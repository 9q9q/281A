"""Run notebook on server and play with position"""
# %%
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from einops import rearrange
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
import numpy.linalg as la
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from typing import Tuple
import pickle
import sys

torch.cuda.set_device(3)
device = 'cuda:'+str(3)

# %%  Define functions

def softknn(
    train_features, train_targets, test_features, test_targets, k=30, T=0.03, 
    max_distance_matrix_size=int(5e6), distance_fx: str = "cosine", 
    epsilon: float = 0.00001) -> Tuple[float]:
    """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
    the weight is computed using the exponential of the temperature scaled cosine
    distance of the samples. If euclidean distance is selected, the weight corresponds
    to the inverse of the euclidean distance.
    k (int, optional): number of neighbors. Defaults to 30.
    T (float, optional): temperature for the exponential. Only used with cosine
        distance. Defaults to 0.03.
    max_distance_matrix_size (int, optional): maximum number of elements in the
        distance matrix. Defaults to 5e6.
    distance_fx (str, optional): Distance function. Defaults to "cosine".
    epsilon (float, optional): Small value for numerical stability. Only used with
        euclidean distance. Defaults to 0.00001.
    
    Adopted from https://github.com/vturrisi/solo-learn
    https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py#L27
    
    Returns:
        Tuple[float]: k-NN accuracy @1 and @5.
    """
    train_features = torch.Tensor(train_features)
    test_features = torch.Tensor(test_features)
    train_targets = torch.Tensor(train_targets)
    test_targets = torch.Tensor(test_targets)

    if distance_fx == "cosine":
        train_features = F.normalize(train_features)
        test_features = F.normalize(test_features)

    num_classes = torch.unique(test_targets).numel()
    num_train_images = train_targets.size(0)
    num_test_images = test_targets.size(0)
    num_train_images = train_targets.size(0)
    chunk_size = min(
        max(1, max_distance_matrix_size // num_train_images),
        num_test_images,
    )
    k = min(k, num_train_images)

    top1, top5, total = 0.0, 0.0, 0
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, chunk_size):
        # get the features for test images
        features = test_features[idx: min(
            (idx + chunk_size), num_test_images), :]
        targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        if distance_fx == "cosine":
            similarities = torch.mm(features, train_features.t())
        elif distance_fx == "euclidean":
            similarities = 1 / \
                (torch.cdist(features, train_features) + epsilon)
        else:
            raise NotImplementedError

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        if distance_fx == "cosine":
            similarities = similarities.clone().div_(T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (
            top5 + correct.narrow(1, 0,
                                  min(5, k, correct.size(-1))).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    return top1, top5


def visualize_grid(img, figsize=(6, 6), colorbar=False, title=""):
    """
    Displaying a list of image in grids. 
    Args:
        features: Image with dimension [b, c, h, w].
        figsize: size of the figure being plotted.
    Returns:
        None
    """
    plt.figure(figsize=figsize)

    grid_h = int(np.sqrt(len(img)))
    imshow = rearrange(img[:(len(img)//grid_h)*grid_h],
                       "(h1 w1) c h w ->  (h1 h) (w1 w) c", h1=grid_h)
    plt.imshow(imshow)
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)

# essential function to implement SMT


def sparsify_general1(x, basis, psi=None, t=0.3):
    """
    This function gives the general sparse feature for image patch x. We calculte the cosine similarity between
    each image patch x and each dictionary element in basis. If the similarity pass threshold t, then the activation
    is 0, else, the actiavtion is 0. 
    
    Assume both x and basis is normalized.
    
    Args:
        x: Flattened image patches with dimension [bsz, p_w*p_h*c].
        basis: dictionary/codebook with dimension [p_w*p_h*c,num_dict_element], each column of the basis is a dictionary element. 
        t: threshold
    Returns:
        bag_of_patche: List of image patches with size [bsz, c, ps, ps, num_patches]. Each patch has the shape [c, ps, ps]
    """

    if psi == None:
        a = (torch.mm(basis.t(), x) > t).float()
    else:
        # a = (basis.T @ x @ psi > t).float()
        a = ((basis.T @ x) + psi > t).float()
    return a


def sparsify_kthresh(x, basis, ts):
    """
    Sparsification strategy in Dasgupta and Tosh 2020. 

    
    Assume both x and basis is normalized.
    
    Args:
        x: Flattened image patches with dimension [bsz, p_w*p_h*c].
        basis: dictionary/codebook with dimension [p_w*p_h*c,num_dict_element], each column of the basis is a dictionary element. 
        t: threshold
    Returns:
        bag_of_patches: List of image patches with size [bsz, c, ps, ps, num_patches]. Each patch has the shape [c, ps, ps]
    """
    a = ((basis.T @ x) > ts).float()
    return a


def unfold_image(imgs, PATCH_SIZE=6, hop_length=2):
    """
    Unfold each image in imgs into a bag of patches.
    Args:
        imgs: Image with dimension [bsz, c, h, w].
        PATCH_SIZE: patch size for each image patch after unfolding, p_h and p_w stands for the height and width of the patch.
    Returns:
        bag_of_patche: List of image patches with size [bsz, c, p_h, p_w, num_patches]. Each patch has the shape [c, p_h, p_w]
    """
    bag_of_patches = F.unfold(imgs, PATCH_SIZE, stride=hop_length)
    bag_of_patches = rearrange(
        bag_of_patches, "bsz (c p_h p_w) b -> bsz c p_h p_w b", c=3, p_h=PATCH_SIZE)
    return bag_of_patches


def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def normalize_patches_rgb(img):
    """Normalize all patches to be between 0 and 1 for visualization only
    img: rgb img of dim [num_patches, c=3, ph, pw]
    axis=(0,2,3) because all patches are from the same image so can normalize 
    the same and want to normalize each channel separately.
    """
    img = img - torch.amin(img, axis=(0, 2, 3), keepdim=True)
    img = img / torch.amax(img, axis=(0, 2, 3), keepdim=True)
    return img


def normalize_batch_rgb(img):
    """Normalize all patches to be between 0 and 1 for visualization only
    img: rgb img of dim [num_patches, c=3, ph, pw]
    axis=(2,3) because all patches are from different images so normalize separately
    """
    img = img - torch.amin(img, axis=(2, 3), keepdim=True)
    img = img / torch.amax(img, axis=(2, 3), keepdim=True)
    return img


def gaussian2d(dim, mean, std):
    """Returns a 2d gaussian of size dim x dim with mean at mean
    dim is scalar, mean is tuple
    """
    x = np.arange(0, dim, 1, float)
    y = x[:, np.newaxis]
    x0 = mean[0]
    y0 = mean[1]
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * std**2))


def visualize_patches(patches, title="", figsize=None, colorbar=False, ncol=None):
    """
    Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches : Tensor of (batch_size, c, h, w).

    title : String; title of figure. Optional.
    """
    size = patches.size(2)
    batch_size = patches.size(0)
    c = patches.size(1)
    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(patches[i], (c, size, size))
        img_grid.append(img)

    if not ncol:
        ncol = int(np.sqrt(batch_size))
    out = make_grid(img_grid, padding=1, nrow=ncol, pad_value=torch.min(patches), scale_each=True)
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    fig = plt.gcf()
    if figsize:
        fig.set_size_inches(figsize)
    # plt.imshow(out, cmap="gray", vmin=-1, vmax=1)  # for gabors
    plt.imshow(out.permute(1, 2, 0))  # for images
    if colorbar:
        plt.colorbar()
#%%
#@title
# Load data from CIFAR10
CIFAR_Path = './cifar-10-batches-py/'
CIFAR_files = []
CIFAR_files.append('data_batch_1')
CIFAR_files.append('data_batch_2')
CIFAR_files.append('data_batch_3')
CIFAR_files.append('data_batch_4')
CIFAR_files.append('data_batch_5')
CIFAR_files_test = []
CIFAR_files_test.append('test_batch')
imgs = torch.zeros([10000 * len(CIFAR_files),32,32,3])
labels = []
for batch_Idx in range(len(CIFAR_files)):
    batch = unpickle(CIFAR_Path + CIFAR_files[batch_Idx])
    data = batch['data']
    for img_idx in range(10000):
        labels.append(batch['labels'][img_idx])
        for color_channel in range(3):
            imgs[img_idx + batch_Idx*10000, :,:,color_channel] = torch.from_numpy( \
                data[img_idx,1024*color_channel:1024*(color_channel+1)].reshape([32,32], order = 'c'))
imgs_test = torch.zeros([10000 * len(CIFAR_files_test),32,32,3])
labels_test = []
for batch_Idx in range(len(CIFAR_files_test)):
    batch = unpickle(CIFAR_Path + CIFAR_files_test[batch_Idx])
    data = batch['data']
    for img_idx in range(10000):
        labels_test.append(batch['labels'][img_idx])
        for color_channel in range(3):
            imgs_test[img_idx + batch_Idx*10000, :,:,color_channel] = torch.from_numpy( \
                data[img_idx,1024*color_channel:1024*(color_channel+1)].reshape([32,32], order = 'c'))

ratio = 10
labels_train = torch.tensor(labels[:len(labels)//ratio])
labels_test = torch.tensor(labels_test[:len(labels_test)//ratio])
imgs_train = imgs[:len(imgs)//ratio].permute(0,3,1,2)
imgs_test = imgs_test[:len(imgs_test)//ratio].permute(0,3,1,2)
imgs_train_reflect = torchvision.transforms.functional.hflip(imgs_train)
imgs_train = torch.cat((imgs_train, imgs_train_reflect))
labels_train = torch.cat((labels_train, labels_train))
del imgs_train_reflect

imgs_train = imgs_train/255.
imgs_test = imgs_test/255.

#%%
# visualize training samples
visualize_grid(imgs_train[:49,:,:,:])

#%%
# Whiten pixel space

BATCH_SIZE = 1000
PATCH_SIZE = 6

# To calculate a covariance matrix in the pixel space for whitening calculation.
V = torch.zeros(PATCH_SIZE**2*3,PATCH_SIZE**2*3).to(device)
BATCH_NUM = 0
for i in tqdm(range(0, imgs_train.size(0), BATCH_SIZE)):
    for j in range(32 - PATCH_SIZE + 1):
        for k in range(32 - PATCH_SIZE + 1):
            patches = imgs_train[i:i+BATCH_SIZE,:,j:j+PATCH_SIZE,k:k + PATCH_SIZE]
            patches = patches.to(device)
            patches = patches.sub(patches.mean((2,3),keepdim =True))
            patches = patches[:,...].reshape(patches.shape[0],-1).t()
            V = V + torch.mm(patches,patches.t())/BATCH_SIZE
            BATCH_NUM +=1
V = V/BATCH_NUM

# Decompose the covariance matrix to obtain the whiten basis. 
w1_c, v1_c = torch.linalg.eig(V + torch.eye(PATCH_SIZE**2*3).to(device) * 1e-7)
w1 = w1_c.real
v1 = v1_c.real
whiteMat = torch.mm((w1.add(1e-3).pow(-0.5)).diag(), v1.t())
colorMat = torch.mm(v1, w1.add(1e-3).pow(0.5).diag())

#%%
# Visualize the whiten basis:
# Because natural iamge statistic is translational invaraint, the whiten basis resemble the fourier basis.
whiten_basis_vis = rearrange(whiteMat,"bsz (c p_h p_w) -> bsz c p_h p_w", p_h=PATCH_SIZE, p_w = PATCH_SIZE)
visualize_grid(whiten_basis_vis.cpu())

#%%
torch.manual_seed(5.23)
BASIS1_NUM = 8192
BASIS1_SIZE = [PATCH_SIZE**2*3, BASIS1_NUM]
basis1 = torch.randn(BASIS1_SIZE).to(device)
basis1_vis = torch.randn(BASIS1_SIZE).to(device)
basis1_vis = torch.randn(BASIS1_SIZE)
#To initialize the dictionary with image patches, we randomly select image patches to be dictionary elements
dict_means = torch.zeros((BASIS1_NUM, 3)).to(device)
for i in tqdm(range(BASIS1_NUM)):
    idx = torch.randint(0,imgs_train.size(0),(1,))
    pos = torch.randint(0,32 - PATCH_SIZE+1,(2,))
    patch = imgs_train[idx[0]:idx[0]+1,:,pos[0]:pos[0]+PATCH_SIZE,pos[1]:pos[1] + PATCH_SIZE]
    patches = patch.to(device) 
    mean = patches.mean((2,3),keepdim =True)
    patch_rm = patches.sub(mean)
    dict_means[i] = mean.reshape(3)
    basis1[:,i] = torch.mm(whiteMat, patch_rm[:,...].reshape(1,-1).t())[:,0]
    basis1_vis[:,i] = patch[:,...].reshape(1,-1).t()[:,0].cpu()
    
basis1 = basis1 + 0.001 * torch.rand(basis1.size(), device = device)
basis1 = basis1.div_(basis1.norm(2,0) + 1e-7)

#%%
dictionary_vis = rearrange(basis1_vis.T[:200],"bsz (c p_h p_w) -> bsz c p_h p_w", p_h=PATCH_SIZE, p_w = PATCH_SIZE)
visualize_patches(dictionary_vis.cpu(), ncol=40, figsize=(20, 10))

#%%
# Intialize all the parameter we need for solving SMT. 

batch_size=10
hop_length = 1
RG = int((32-PATCH_SIZE)/hop_length)+1
threshold=0.3

basis1 = basis1.to(device)
whiteMat = whiteMat.to(device)
#To build generalized SMT:
BASIS1_NUM = basis1.size(1)
coStat = torch.zeros(BASIS1_NUM,BASIS1_NUM, device=device)
VStat = torch.zeros(BASIS1_NUM,BASIS1_NUM, device=device)
coStat.fill_(0)
VStat.fill_(0)

patches = torch.zeros(RG**2,3,PATCH_SIZE,PATCH_SIZE)
temp2 = torch.zeros(BASIS1_NUM, batch_size * RG**2, device=device)

#%% 
# randomly draw position encoding vector with element either -1 or 1 with equal probability
# psi = torch.randint(0,2,(RG, RG), device=device)*2-1
# psi = psi.flatten().expand((BASIS1_NUM, batch_size, RG**2)).reshape(
    # BASIS1_NUM, -1)

# gaussian - low accuracy
# get a 2d gaussian at each position in (RG, RG)
# psi = torch.zeros(RG**2, RG**2, device=device)
# # generate list of indices as tuples for 27x27 grid
# grid_idx = [(i,j) for i in range(RG) for j in range(RG)]
# std = 2
# for i, mu in enumerate(grid_idx):
#     # 2d gaussian with mean mu, std 1
#     gaussian = gaussian2d(RG, mu, std)
#     psi[:,i] = torch.from_numpy(gaussian.flatten())
# plt.imshow(psi.cpu()); plt.show()
# psi = psi.expand((batch_size, batch_size, RG**2, RG**2)).permute(0, 2, 1, 3).reshape(
#     batch_size*RG**2, batch_size*RG**2)

# sinusoid like in attention paper, same dim as data (num_patches x patch_size)
# psi = torch.zeros((1, RG**2, PATCH_SIZE**2)).to(device)
# X = torch.arange(RG**2, dtype=torch.float32).reshape(
#     -1, 1) / torch.pow(10000, torch.arange(
#     0, PATCH_SIZE**2, 2, dtype=torch.float32) / PATCH_SIZE**2)
# psi[:, :, 0::2] = torch.sin(X)
# psi[:, :, 1::2] = torch.cos(X)
# psi = rearrange(psi.expand((batch_size, 3, RG**2, PATCH_SIZE**2)), "bs c n_p n_pix -> (n_pix c) (bs n_p)")

# add after encoding, before sparsification
psi = torch.zeros((1, BASIS1_NUM, RG**2+1)).to(device)
X = torch.arange(BASIS1_NUM, dtype=torch.float32).reshape(
    -1, 1) / torch.pow(10000, torch.arange(
    0, RG**2, 2, dtype=torch.float32) / RG**2)
psi[:, :, 0::2] = torch.sin(X)
psi[:, :, 1::2] = torch.cos(X)
psi = rearrange(psi[:,:,-1:].expand((batch_size, BASIS1_NUM, RG**2)), "bs n_b n_p -> n_b (bs n_p)")

#%% this takes 4 mins on gpu
# Training (collecting co-variance and co-occurence)
sparsity = []
torch.manual_seed(5.23)
ts = torch.randn(BASIS1_NUM, 1).clip(min=0).to(device)
for idx in tqdm(range(0,imgs_train.size(0), batch_size)):
#     unfold each image into a bag of patches
    patches =unfold_image(imgs_train[idx:idx+batch_size].to(device),PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
#     demean/center each image patch
    patches = patches.sub(patches.mean((2,3),keepdim =True))
#     aggregate all patches into together (squeeze into one dimension).
    x = rearrange(patches,"bsz c p_h p_w b  ->bsz (c p_h p_w) b ")
    x_flat = rearrange(x,"bsz p_d hw -> p_d (bsz hw)")
#     apply whiten transform to each image patch
    x_flat = torch.mm(whiteMat, x_flat)
#     normalize each image patch
    x_flat = x_flat.div(x_flat.norm(dim = 0, keepdim=True)+1e-9)
#     extract sparse feature vector ahat from each image patch, sparsify_general1 is f_gq in the paper
    # ahat = sparsify_general1(x_flat, basis1, t=threshold)
    ahat = sparsify_kthresh(x_flat, basis1, ts)
    sparsity.append(ahat.mean().item())
    # position encoding (bind)
    # ahat *= psi
    # "attention matrix"
    # ahat = ahat @ psi
#     Average sparse feacture over current context window
    ahat_bar = rearrange(ahat,"c (b hw) -> c b hw",b=batch_size).sum(-1)
#     update co-occurence statistic
    temp = torch.mm(ahat_bar, ahat_bar.t())
    coStat.add_(temp); del temp; torch.cuda.empty_cache()
#     update co-variance statistic
    temp = torch.mm(ahat, ahat.t())
    VStat.add_(temp); del temp; torch.cuda.empty_cache()
del ahat, x; 
torch.cuda.empty_cache()

#%%
coStat = torch.load("coStat.pt")
VStat = torch.load("VStat.pt")

#%%
# move off gpu, free cache
# VStat_c = VStat.cpu(); coStat_c = coStat.cpu(); del VStat, coStat; torch.cuda.empty_cache()
# The following derives the SMT embedding for the feature function:
ADDA = RG**2 * VStat - coStat
wV, vV = torch.linalg.eigh(VStat, UPLO='U')
wV = wV.clamp_(min=0.000001)
V_invhalf = torch.mm(vV, torch.mm(wV.pow(-0.5).diag(),vV.t()))
Q = torch.mm(V_invhalf, torch.mm(ADDA, V_invhalf))
wQ, vQ = torch.linalg.eigh(Q, UPLO='U')

#%%
num_dim = 512
U = vQ[:,8:8+num_dim]
P_star = torch.mm(V_invhalf, U).t().to(device)

#%%
# # Manifold learning without temperal constraint
# num_dim = 4096
# P_star = vV[:,:num_dim].T.to(device)

# P_star = torch.eye(8192,8192).to(device)
# P_star = vV[:,:].T.to(device)
# num_dim = P_star.shape[1]

U = vQ[:,:]
P_star_full = torch.mm(V_invhalf, U).t().to(device)

#%%
# patial pooling to aggregate patch embedding in to image embedding
output_w = 3
temp_train_1 = torch.zeros([imgs_train.size(0),num_dim,output_w,output_w])
temp_test_1 = torch.zeros([imgs_test.size(0),num_dim,output_w,output_w])
temp_train_1.fill_(0)
temp_test_1.fill_(0)

# for HD style, would "pool" by bundling all vects into one?
# temp_train_1 = torch.zeros([imgs_train.size(0),num_dim])
# temp_test_1 = torch.zeros([imgs_test.size(0),num_dim])

# for epoch in range(4):
for idx in tqdm(range(0,imgs_train.size(0), batch_size)):
#     unfold each image into a bag of patches
    patches =unfold_image(imgs_train[idx:idx+batch_size].to(device),PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
#     demean/center each image patch
    patches = patches.sub(patches.mean((2,3),keepdim =True))
#     aggregate all patches into together (squeeze into one dimension).
    x = rearrange(patches,"bsz c p_h p_w b  ->bsz (c p_h p_w) b ")
    x_flat = rearrange(x,"bsz p_d hw -> p_d (bsz hw)")
#     apply whiten transform to each image patch
    x_flat = torch.mm(whiteMat, x_flat)
#     normalize each image patch
    x_flat = x_flat.div(x_flat.norm(dim = 0, keepdim=True)+1e-9)
#     extract sparse feature vector ahat from each image patch, sparsify_general1 is f_gq in the paper
    ahat = sparsify_general1(x_flat, basis1, t=threshold)
    # ahat *= psi
    # ahat = ahat.reshape(BASIS1_NUM, batch_size, RG**2).sum(dim=-1)
#     project the sparse code into the spectral embeddings
    temp = torch.mm(P_star, ahat)
    temp = temp.div(temp.norm(dim=0, keepdim=True)+ 1e-9)
    temp = rearrange(temp,"c (b2 h w) -> b2 c h w",b2=batch_size,h=RG)
#     apply spatial pooling
# TODO pool by summing in HD way?
    temp_train_1[idx:idx+batch_size,...] = F.adaptive_avg_pool2d(F.avg_pool2d(temp, kernel_size = 5, stride = 3), output_w)


temp_test_unpooled = torch.zeros([imgs_test.size(0),num_dim,RG,RG])
for idx in tqdm(range(0,imgs_test.size(0), batch_size)):
#     unfold each image into a bag of patches
    patches =unfold_image(imgs_test[idx:idx+batch_size].to(device),PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
#     demean/center each image patch
    patches = patches.sub(patches.mean((2,3),keepdim =True))
#     aggregate all patches into together (squeeze into one dimension).
    x = rearrange(patches,"bsz c p_h p_w b  ->bsz (c p_h p_w) b ")
    x_flat = rearrange(x,"bsz p_d hw -> p_d (bsz hw)")
#     apply whiten transform to each image patch
    x_flat = torch.mm(whiteMat, x_flat)
#     normalize each image patch
    x_flat = x_flat.div(x_flat.norm(dim = 0, keepdim=True)+1e-9)
#     extract sparse feature vector ahat from each image patch, sparsify_general1 is f_gq in the paper
    ahat = sparsify_general1(x_flat, basis1, t=threshold)
    # ahat = ahat @ psi
    # ahat *= psi
    # ahat = ahat.reshape(BASIS1_NUM, batch_size, RG**2).sum(dim=-1)
#     project the sparse code into the spectral embeddings
    temp = torch.mm(P_star, ahat)
    temp = temp.div(temp.norm(dim=0, keepdim=True)+ 1e-9)
    temp = rearrange(temp,"c (b2 h w) -> b2 c h w",b2=batch_size,h=RG)
#     apply spatial pooling
# TODO pool by summing in HD way?
    temp_test_1[idx:idx+batch_size,...] = F.adaptive_avg_pool2d(F.avg_pool2d(temp, kernel_size = 5, stride = 3), output_w)
    # temp_test_unpooled[idx:idx+batch_size,...] = temp

# worse if not normalized
temp_test_unpooled = temp_test_unpooled.div(temp_test_unpooled.norm(dim=(-3), keepdim=True) + 1e-9)
temp_train_1 = temp_train_1.div(temp_train_1.norm(dim=(-3), keepdim=True) + 1e-9)
temp_test_1 = temp_test_1.div(temp_test_1.norm(dim=(-3), keepdim=True) + 1e-9)
# temp_train_1 = temp_train_1.div(temp_train_1.norm(dim=(-1), keepdim=True) + 1e-9)
# temp_test_1 = temp_test_1.div(temp_test_1.norm(dim=(-1), keepdim=True) + 1e-9)

#%%
# temp_train_1 = torch.load("beta_train_8192.pt")
# temp_test_1 = torch.load("beta_test_8192.pt")

temp_train_1 = torch.load("beta_train_512.pt")
temp_test_1 = torch.load("beta_test_512.pt")

# torch.save(temp_train_1, "beta_train_4096.pt")
# torch.save(temp_test_1, "beta_test_4096.pt")

#%%
X = temp_train_1.flatten(start_dim = 1,end_dim = -1).numpy()
X_test = temp_test_1.flatten(start_dim = 1,end_dim = -1).numpy()
# X  = temp_train_1.mean(dim = [-1,-2]).numpy()
# X_test  = temp_test_1.mean(dim = [-1,-2]).numpy()
y = labels_train
y_test = labels_test

#%%
# if resulting vectors are HD (have been bound with pos), then can use other methods to classify, e.g. clustering? look at lit
clf = LogisticRegression(random_state=0)
clf.fit(X,y) 
y_test_hat = clf.predict(X_test)
acc_score = accuracy_score(y_test,y_test_hat)
print(acc_score)

#%% run knn
top1, top5 = softknn(X, y, X_test, y_test, k=30, T=0.03)
print(f"Top-1 acc: {top1:.2f}, Top-5 acc: {top5:.2f}")

#%% run kmeans
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
y_test_hat = kmeans.predict(X_test)
#%% visualize clusters
plt.close()
for i in range(n_clusters):
    labels_idx = np.where(y_test_hat == i)[0]
    visualize_grid(imgs_test[labels_idx,:,:,:])

#%% visualize slow components

# project into pixel space
num_dim = 512
P_star_inv = torch.load("P_star_full_inv.pt", map_location=torch.device(device))
# P_star_inv = torch.load("P_star_inv.pt", map_location=torch.device(device))
P_proj = (basis1 @ P_star_inv)

# unwhiten
colorMat = colorMat.cpu(); P_proj = P_proj.cpu()
unwhitened = colorMat @ P_proj
# unwhitened = unwhitened.div_(unwhitened.norm(p=2, dim=0, keepdim=True) + 1e-10)

P_proj_vis = rearrange(unwhitened,"(c p_h p_w) n_p -> n_p c p_h p_w", 
                       p_h=PATCH_SIZE, p_w = PATCH_SIZE, n_p = BASIS1_NUM)
P_proj_vis += dict_means.cpu().unsqueeze(-1).unsqueeze(-1)  # add back mean
P_proj_vis = P_proj_vis.div_(P_proj_vis.norm(p=2, dim=(2, 3), keepdim=True) + 1e-7)
# visualize_patches(normalize_patches_rgb(P_proj_vis[:225]), figsize=(10,10))
visualize_patches(normalize_batch_rgb(P_proj_vis[8:8+200]), figsize=(10, 8), ncol=20)
# visualize_patches(P_proj_vis[8:8+225], figsize=(10,10))
plt.show()

#%%
beta_batch = torch.load("beta_batch.pt")
ahat = torch.load("ahat.pt")
temp = torch.load("temp.pt")
temp_test_unpooled = torch.load("temp_test_unpooled.pt")

#%%
batch_size = 1000
output_w = 3
patches = unfold_image(imgs_test[:batch_size],
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
#     demean/center each image patch
patches = patches.sub(patches.mean((2,3),keepdim =True))
#     aggregate all patches into together (squeeze into one dimension).
x = rearrange(patches,"bsz c p_h p_w b  ->bsz (c p_h p_w) b ")
x_flat = rearrange(x,"bsz p_d hw -> p_d (bsz hw)")
#     apply whiten transform to each image patch
x_flat = torch.mm(whiteMat.cpu(), x_flat)
#     normalize each image patch
x_flat = x_flat.div(x_flat.norm(dim = 0, keepdim=True)+1e-9)
#     extract sparse feature vector ahat from each image patch, sparsify_general1 is f_gq in the paper
ahat = sparsify_general1(x_flat, basis1.cpu(), t=threshold)
# ahat = ahat @ psi
# ahat *= psi
# ahat = ahat.reshape(BASIS1_NUM, batch_size, RG**2).sum(dim=-1)
#     project the sparse code into the spectral embeddings
temp = torch.mm(P_star.cpu(), ahat.cpu())
temp = temp.div(temp.norm(dim=0, keepdim=True)+ 1e-9)
temp = rearrange(temp,"c (b2 h w) -> b2 c h w",b2=batch_size,h=RG)
#     apply spatial pooling
# TODO pool by summing in HD way?
beta_batch = F.adaptive_avg_pool2d(F.avg_pool2d(temp, kernel_size = 5, stride = 3), output_w)
ahat = rearrange(ahat, "n_b (bs n_p) -> bs n_p n_b", bs=batch_size, n_b=BASIS1_NUM, 
                 n_p=RG**2).cpu()

#%% for test img 1, look at beta and look at corresponding P in pixel space
# one beta is 512 x 3 x 3, so average to 512 vector where each element corresponds to a row of P
img_id = 0
patch_id = 50
one_patch = patches[img_id, :, :, :, patch_id]
temp = temp.cpu()
# temp_test_unpooled = temp_test_unpooled.cpu()
patches = unfold_image(imgs_test[:batch_size].cpu(),
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
one_beta = temp[img_id].reshape(num_dim, RG**2)[:,patch_id]
# one_beta = temp_test_unpooled[img_id].reshape(num_dim, RG**2)[:,patch_id]
# show whole image
visualize_grid(imgs_test[img_id].unsqueeze(0))
# show one patch
visualize_grid(one_patch.unsqueeze(0)); plt.show()
# show rows of Ps with highest activations for this image (abs value)
beta_sorted, beta_sorted_id = torch.sort(one_beta.abs(), descending=True)
P_sorted = P_proj_vis[beta_sorted_id] 
# visualize_grid(P_sorted[:25])  # look at top 10
visualize_patches(normalize_batch_rgb(P_sorted[:36]), title="P corresponding to top values of beta")  # look at top 10

# show highest activation phi for this image
torch.manual_seed(0)
one_ahat = ahat[img_id, patch_id]
alpha_ones_id = torch.where(one_ahat == 1)[0]
alpha_ones_id = alpha_ones_id[torch.randperm(len(alpha_ones_id))]  # take random ones
basis1_sorted = basis1_vis[:, alpha_ones_id].cpu()
basis1_sorted_vis = normalize_patches_rgb(rearrange(basis1_sorted[:,:25], 
                                          "(c p_h p_w) n_p -> n_p c p_h p_w", c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE))
visualize_patches(basis1_sorted_vis, title="phi corresponding to activated alpha")

#%% show phis with highest weights corresponding to a given P
# P_id = beta_sorted_id[9]
P_id = 0
one_P = P_star[P_id]
sorted_P_id = torch.argsort(one_P, descending=True)
basis1_sorted = basis1_vis[:, sorted_P_id].cpu()
basis1_sorted_vis = normalize_patches_rgb(rearrange(basis1_sorted[:,:100], 
                                          "(c p_h p_w) n_p -> n_p c p_h p_w", c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE))
visualize_grid(normalize_patches_rgb(P_proj_vis[P_id].unsqueeze(0)), title=f"P{P_id}")
visualize_grid(basis1_sorted_vis, title=f"{basis1_sorted_vis.shape[0]} closest phis to P{P_id}")

#%% for a given row P_i, look for images with highest beta_i activation 
P_id = 0
beta_mean = beta_batch.mean(axis=(-1,-2))
# beta_mean = temp_test_1.mean(axis=(-1,-2))
beta_sorted, beta_sorted_id = torch.sort(beta_mean[:,P_id], descending=True)
imgs_sorted = imgs_test[beta_sorted_id] 
visualize_patches(imgs_sorted[:100], title=f"images with highest beta activation for P{P_id}")

#%% for a given row P_i, look for patches with highest beta_i activation 
plt.close()
P_id = 9
batch_size=1000
patches = unfold_image(imgs_test[:batch_size].cpu(),
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
visualize_grid(normalize_batch_rgb(P_proj_vis[P_id+8].unsqueeze(0)), title=f"P{P_id}")
plt.show()
temp_reshaped = rearrange(temp, "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size, n_h=RG, n_w=RG)
# temp_reshaped = rearrange(temp_test_unpooled, "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size, n_h=RG, n_w=RG)
patches_reshaped = rearrange(patches, "bs c p_h p_w n_p -> (bs n_p) c p_h p_w", bs=batch_size, c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE)
temp_sorted, temp_sorted_id = torch.sort(temp_reshaped[:,P_id], descending=True)
patches_sorted = patches_reshaped[temp_sorted_id] 
bin_size = 400
# for i in range(10):
#     visualize_patches(patches_sorted[i*bin_size:i*bin_size+bin_size], 
#                   title=f"patches in bin {i} for P{P_id}")
#     plt.show()

i = 700
visualize_patches(patches_sorted[i*bin_size:i*bin_size+bin_size], 
                  title=f"patches in bin {i} for P{P_id}")

#%% look at neighbors of patches (fig 4 iclr paper)
batch_size=1000
patches = unfold_image(imgs_test[:batch_size].cpu(),
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
n_neighbors = 200
X = rearrange(temp, "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size, n_h=RG, n_w=RG)
# X = rearrange(temp_test_unpooled, "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size, n_h=RG, n_w=RG)
nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(X)
#%%
img_id = 821  # 0 to 999
patch_id = 621 # 0 to 728
one_patch = patches[img_id, :, :, :, patch_id]
neigh_dist, neigh_id = nn.kneighbors(temp.reshape(batch_size, num_dim, RG**2)[img_id, :, patch_id].unsqueeze(0))
# neigh_dist, neigh_id = nn.kneighbors(temp_test_unpooled.reshape(batch_size, num_dim, RG**2)[img_id, :, patch_id].unsqueeze(0))
patches_reshaped = rearrange(patches, "bs c p_h p_w n_p -> (bs n_p) c p_h p_w", bs=batch_size, c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE)
sorted_patches = patches_reshaped[neigh_id]
visualize_patches(sorted_patches[:128], title=f"neighbors of img{img_id} patch{patch_id}", ncol=16)
plt.show()

#%% steering in embedding space doesn't work because quantization
torch.manual_seed(0)
temp_reshaped = rearrange(temp, "bs f n_h n_w -> bs f (n_h n_w)")
lambd = 0
beta0 = temp_reshaped[0, :, 1]
beta1 = temp_reshaped[0, :, 2]

beta = lambd * beta0 + (1-lambd) * beta1
# beta1 = temp_reshaped[7, :, 400]
# beta_avg = (beta0+beta1)/2
P_rand = (torch.randn(BASIS1_NUM, BASIS1_NUM).to(device) + P_star_inv.mean()) * P_star_inv.std()
# alpha = (P_star_inv[:num_dim].T @ beta.to(device)>0.3)*1.  # try other dim?
alpha = (P_rand[:num_dim].T @ beta.to(device)>0.3)*1.  # try other dim?
recon = basis1 @ alpha
recon_re = rearrange(colorMat.to(device)@recon, "(b c p_h p_w) -> b c p_h p_w", b=1, c=3, p_h=6, p_w=6)
visualize_patches(normalize_batch_rgb(recon_re.cpu()), title=f"λ={lambd} (unpooled β0 projected to pixel space)")
# visualize_patches(recon_re.cpu())

#%% what about opposite - change patch position slightly and see what happens to beta
plt.close()
img_id = 0
patch_id = 0
lambd = 1
threshold = 1
patch1 = patches[0, :, :, :, patch_id] 
patch0 = patches[1, :, :, :, patch_id]
# subtract mean
patch0 = patch0.sub(patch0.mean((1,2),keepdim =True))
patch1 = patch1.sub(patch1.mean((1,2),keepdim =True))
patch = lambd * patch0 + (1-lambd) * patch1 
visualize_patches(normalize_batch_rgb(torch.stack((patch0, patch1))))
plt.show()
patch = whiteMat @ patch.reshape(108, 1)
patch = patch.div(patch.norm(dim = 0, keepdim=True)+1e-9)
ahat = sparsify_general1(patch, basis1.cpu(), t=threshold)

beta = torch.mm(P_star.cpu(), ahat.cpu())
beta = beta.div(beta.norm(dim=0, keepdim=True)+ 1e-9)
plt.stem(beta.cpu().numpy()); plt.show()


#%% don't include image of the original patch. but still includes multiple patches from same image for different images
# img_id = 0  # 0 to 999
# patch_id = 15  # 0 to 728
# one_patch = patches[img_id, :, :, :, patch_id]
# # only look at patches not in this image:
# X_minus_img = rearrange(temp[np.arange(temp.shape[0])!=img_id], "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size-1, n_h=RG, n_w=RG)
# nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(X_minus_img)
# neigh_dist, neigh_id = nn.kneighbors(temp.reshape(batch_size, num_dim, RG**2)[img_id, :, patch_id].unsqueeze(0))
# # only consider patches not in this image:
# patches_reshaped = rearrange(patches[np.arange(patches.shape[0])!=img_id], "bs c p_h p_w n_p -> (bs n_p) c p_h p_w", bs=batch_size-1, c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE)
# sorted_patches = patches_reshaped[neigh_id]
# # sorted_patches = sorted_patches.div_(sorted_patches.norm(dim=(-1, -2), keepdim=True) + 1e-7)
# visualize_patches(normalize_patches_rgb(sorted_patches[:100]), title=f"{100} neighbors of img{img_id} patch{patch_id}")
# plt.show()

#%%
# grab some patches and find the ones with highest activation for each slow component
# cluster in ahat vs look at P?
# or reconstruct by solving for alpha_rec
# or look at figure 4 in minimalistic paper
