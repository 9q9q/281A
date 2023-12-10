"""Some copied from smt_general_sparsity, experimenting with positional encoding."""

#%%
import sys
import pickle
from typing import Tuple
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import imageio
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from einops import rearrange

torch.cuda.set_device(0)
device = 'cuda:'+str(0)

#%%  Define functions
def softknn(train_features,train_targets,test_features,test_targets,k=30,T=0.03,max_distance_matrix_size=int(5e6),distance_fx: str = "cosine",epsilon: float = 0.00001) -> Tuple[float]:
    """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
    the weight is computed using the exponential of the temperature scaled cosine
    distance of the samples. If euclidean distance is selected, the weight corresponds
    to the inverse of the euclidean distance.
    
    Adopted from https://github.com/vturrisi/solo-learn
    
    Returns:
        Tuple[float]: k-NN accuracy @1 and @5.
    """

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
        features = test_features[idx : min((idx + chunk_size), num_test_images), :]
        targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        if distance_fx == "cosine":
            similarities = torch.mm(features, train_features.t())
        elif distance_fx == "euclidean":
            similarities = 1 / (torch.cdist(features, train_features) + epsilon)
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
            top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    return top1, top5
    
def visualize_grid(img,figsize=(6,6), colorbar=False, title=""):
    """
    Displaying a list of image in grids. 
    Args:
        features: Image with dimension [b, c, h, w].
        figsize: size of the figure being plotted.
    Returns:
        None
    """
    plt.figure(figsize = figsize)
    
    grid_h = int(np.sqrt(len(img)))
    imshow  = rearrange(img[:(len(img)//grid_h)*grid_h],"(h1 w1) c h w ->  (h1 h) (w1 w) c",h1=grid_h)
    plt.imshow(imshow)
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)
    
# essential function to implement SMT
def sparsify_general1(x, basis, t = 0.3):
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

def unfold_image(imgs,PATCH_SIZE=6,hop_length=2):
    """
    Unfold each image in imgs into a bag of patches.
    Args:
        imgs: Image with dimension [bsz, c, h, w].
        PATCH_SIZE: patch size for each image patch after unfolding, p_h and p_w stands for the height and width of the patch.
    Returns:
        bag_of_patche: List of image patches with size [bsz, c, p_h, p_w, num_patches]. Each patch has the shape [c, p_h, p_w]
    """
    bag_of_patches = F.unfold(imgs, PATCH_SIZE, stride=hop_length)
    bag_of_patches = rearrange(bag_of_patches,"bsz (c p_h p_w) b -> bsz c p_h p_w b",c=3,p_h=PATCH_SIZE)
    return bag_of_patches

def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict    

#%% Load data from CIFAR10
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

# visualize training samples
visualize_grid(imgs_train[:49,:,:,:])

#%% # Whiten pixel space

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

# Visualize the whiten basis:
# Because natural iamge statistic is translational invaraint, the whiten basis resemble the fourier basis.
whiten_basis_vis = rearrange(whiteMat,"bsz (c p_h p_w) -> bsz c p_h p_w", p_h=PATCH_SIZE, p_w = PATCH_SIZE)
visualize_grid(whiten_basis_vis.cpu())

#%% create dict
torch.manual_seed(5.23)
BASIS1_NUM = 8192
BASIS1_SIZE = [PATCH_SIZE**2*3, BASIS1_NUM]
basis1 = torch.randn(BASIS1_SIZE).to(device)
basis1_vis = torch.randn(BASIS1_SIZE).to(device)
# basis1_vis = torch.randn(BASIS1_SIZE)
#To initialize the dictionary with image patches, we randomly select image patches to be dictionary elements
for i in tqdm(range(BASIS1_NUM)):
    idx = torch.randint(0,imgs_train.size(0),(1,))
    pos = torch.randint(0,32 - PATCH_SIZE+1,(2,))
    patch = imgs_train[idx[0]:idx[0]+1,:,pos[0]:pos[0]+PATCH_SIZE,pos[1]:pos[1] + PATCH_SIZE]
    patches = patch.to(device) 
    patch_rm = patches.sub(patches.mean((2,3),keepdim =True))
    basis1[:,i] = torch.mm(whiteMat, patch_rm[:,...].reshape(1,-1).t())[:,0]
    basis1_vis[:,i] = patch[:,...].reshape(1,-1).t()[:,0].cpu()
    
basis1 = basis1 + 0.001 * torch.rand(basis1.size(), device = device)
basis1 = basis1.div_(basis1.norm(2,0) + 1e-7)

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
temp1 = torch.zeros(BASIS1_NUM, batch_size, device=device)
temp2 = torch.zeros(BASIS1_NUM, batch_size * RG**2, device=device)

#%%  play with one img
idx =2
#     unfold each image into a bag of patches
patches =unfold_image(imgs_train[idx:idx+batch_size].to(device),PATCH_SIZE=PATCH_SIZE,hop_length=hop_length)
#     demean/center each image patch
patches = patches.sub(patches.mean((2,3),keepdim =True))
#     aggregate all patches into together (squeeze into one dimension).
x = rearrange(patches,"bsz c p_h p_w b  ->bsz (c p_h p_w) b ")  # bs x num_filters x num_patches_per_img
one_x = x[idx]
x_flat = rearrange(one_x,"p_d hw -> p_d hw")  # now num_filters x num_patches_per_img
#     apply whiten transform to each image patch
x_flat = torch.mm(whiteMat, x_flat)
#     normalize each image patch
x_flat = x_flat.div(x_flat.norm(dim = 0, keepdim=True)+1e-9)

y = (basis1.T @ x_flat).float()  # num_filters x num_patches_per_img

y_vis = rearrange(y, "b (c p_h p_w) -> b c p_h p_w", c=1, p_h=RG)
visualize_grid(y_vis.cpu()[:100])

#%% contruct psi, num_patches_per_img x num_patches_per_img
# each column is a 1d gaussian centered on the diagonal
sigma = 1
psi = torch.zeros(RG**2, RG**2, device=device)
for i in range(RG**2):
    # 1d gaussian function centered on the diagonal
    psi[:,i] = torch.exp(-(torch.arange(RG**2) - i)**2 / (2 * sigma**2))

plt.imshow(psi.cpu()[:100,:100])    
plt.show()

#%% positional encoding?
A = y @ psi
# normalize A
# A = A.div(A.norm(dim = 0, keepdim=True)+1e-9)
A_vis = rearrange(A, "b (c p_h p_w) -> b c p_h p_w", c=1, p_h=RG)
visualize_grid(A_vis.cpu()[:100])
visualize_grid(y_vis.cpu()[:100])

#%% project back to image space?
A_proj = basis1 @ A
A_proj_vis = rearrange(A_proj.T,"bsz (c p_h p_w) -> bsz c p_h p_w", p_h=PATCH_SIZE, p_w = PATCH_SIZE,c=3)
visualize_grid(A_proj_vis.cpu(), title="projected y with positional encoding")

y_proj = basis1 @ y
y_proj_vis = rearrange(y_proj.T,"bsz (c p_h p_w) -> bsz c p_h p_w", p_h=PATCH_SIZE, p_w = PATCH_SIZE,c=3)
visualize_grid(y_proj_vis.cpu(), title="projected y without positional encoding")

#%% vis one x
one_x_vis = rearrange(one_x, "(c p_h p_w) n -> n c p_h p_w", c=3, p_h=PATCH_SIZE, p_w=PATCH_SIZE)
visualize_grid(one_x_vis.cpu(), title="one x patchified", colorbar=True)

