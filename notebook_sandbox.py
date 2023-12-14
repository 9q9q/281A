"""Run notebook on server and play with position"""
# %%
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.cluster import KMeans
from einops import rearrange
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision
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


def sparsify_general1(x, basis, t=0.3):
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

    a = (torch.mm(basis.t(), x) > t).float()
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

def gaussian2d(dim, mean, std):
    """Returns a 2d gaussian of size dim x dim with mean at mean
    dim is scalar, mean is tuple
    """
    x = np.arange(0, dim, 1, float)
    y = x[:, np.newaxis]
    x0 = mean[0]
    y0 = mean[1]
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * std**2))
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
visualize_grid(dictionary_vis.cpu())

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

# sinusoid like in attention paper


#%% this takes 4 mins on gpu
# Training (collecting co-variance and co-occurence)
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

#%%
# patial pooling to aggregate patch embedding in to image embedding
output_w = 3
temp_train_1 = torch.zeros([imgs_train.size(0),num_dim,output_w,output_w])
temp_test_1 = torch.zeros([imgs_test.size(0),num_dim,output_w,output_w])

# for HD style, would "pool" by bundling all vects into one?
# temp_train_1 = torch.zeros([imgs_train.size(0),num_dim])
# temp_test_1 = torch.zeros([imgs_test.size(0),num_dim])
temp_train_1.fill_(0)
temp_test_1.fill_(0)

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
    # temp_train_1[idx:idx+batch_size,...] = temp.mean(dim=(-1,-2))


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
    # temp_test_1[idx:idx+batch_size,...] = temp.mean(dim=(-1,-2))

# worse if not normalized
# temp_train_1 = temp_train_1.div(temp_train_1.norm(dim=(-3), keepdim=True) + 1e-9)
# temp_test_1 = temp_test_1.div(temp_test_1.norm(dim=(-3), keepdim=True) + 1e-9)
temp_train_1 = temp_train_1.div(temp_train_1.norm(dim=(-1), keepdim=True) + 1e-9)
temp_test_1 = temp_test_1.div(temp_test_1.norm(dim=(-1), keepdim=True) + 1e-9)

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
P_star_inv = torch.load("P_star_inv.pt", map_location=torch.device(device))
P_proj = basis1 @ P_star_inv

# unwhiten
colorMat = colorMat.cpu(); P_proj = P_proj.cpu()
unwhitened = colorMat @ P_proj

P_proj_vis = rearrange(unwhitened,"(c p_h p_w) n_p -> n_p c p_h p_w", 
                       p_h=PATCH_SIZE, p_w = PATCH_SIZE, n_p = BASIS1_NUM)
P_proj_vis = P_proj_vis.div_(P_proj_vis.norm(2,0) + 1e-7) * 255
P_proj_vis = normalize_patches_rgb(P_proj_vis)[:num_dim]
visualize_grid(P_proj_vis)


#%%
batch_size = 10
output_w = 3
patches =unfold_image(imgs_test[:batch_size].to(device),
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
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
beta_batch = F.adaptive_avg_pool2d(F.avg_pool2d(temp, kernel_size = 5, stride = 3), output_w)
ahat = rearrange(ahat, "n_b (bs n_p) -> bs n_p n_b", bs=batch_size, n_b=BASIS1_NUM, 
                 n_p=RG**2).cpu()

#%% for test img 1, look at beta and look at corresponding P in pixel space
# one beta is 512 x 3 x 3, so average to 512 vector where each element corresponds to a row of P
img_id = 0
beta_id = 15
temp = temp.cpu()
patches = unfold_image(imgs_test[:batch_size].cpu(),
                      PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
one_beta = temp[img_id].reshape(num_dim, RG**2)[:,beta_id]
one_patch = patches[img_id, :, :, :, beta_id]

# show whole image
# visualize_grid(imgs_test[0].unsqueeze(0))
# show one patch
visualize_grid(one_patch.unsqueeze(0)); plt.show()
# show rows of Ps with highest activations (abs value)
beta_sorted, beta_sorted_id = torch.sort(one_beta.abs(), descending=True)
P_sorted = P_proj_vis[beta_sorted_id] 
visualize_grid(P_sorted[:10])  # look at top 10

# show highest activation phi
one_ahat = ahat[img_id, beta_id]
alpha_sorted, alpha_sorted_id = torch.sort(one_ahat.abs(), descending=True)
basis1_sorted = basis1[:, alpha_sorted_id]
visualize_grid(basis1_sorted[:10])


#%%
# grab some patches and find the ones with highest activation for each slow component
# or reconstruct by solving for alpha_rec
# or look at figure 4 in minimalistic paper