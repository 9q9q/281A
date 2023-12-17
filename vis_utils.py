"""Utils for visualization."""

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


def show_P(P_inv: torch.Tensor, phi: torch.Tensor, colorMat: torch.Tensor, 
            patch_size: int = 6, title: str = None, n_patches: int = 200, 
            ncol: int = 20, start_at: int = 8, 
            figsize: Tuple[int, int] = (10, 10)):
    """Plot P projected into pixel space. Note that P must be the full 
    square matrix.

    Args:
        P_inv: torch.Tensor, shape (basis_num, basis_num)
        phi: torch.Tensor, shape (basis_num, patch_size**2*num_channels)
        colorMat: unwhitening matrix, shape (patch_size**2*num_channels, 
                  patch_size**2*num_channels)
        title: str (optional)
        n_patches: num patches to show, int (optional)
        ncol: num cols, int (optional)
        start_at: how many of the first P to ignore (often too slow), int (optional)
        figsize: Tuple[int, int] (optional)

    Example call:
        P_inv = torch.linalg.inv(P)  # P is full square matrix
        vis_utils.show_P(P_star_inv, basis1, colorMat, ncol=20, n_patches=200)
    """

    basis_num = phi.shape[1]
    P_proj = phi @ P_inv
    unwhitened = colorMat @ P_proj
    P_proj_vis = rearrange(unwhitened.cpu(),"(c p_h p_w) n_p -> n_p c p_h p_w", 
                       p_h=patch_size, p_w = patch_size, n_p = basis_num, c=3)
    P_proj_vis = P_proj_vis.div_(P_proj_vis.norm(p=2, dim=(2, 3), 
                                 keepdim=True) + 1e-7)
    visualize_patches(normalize_batch_rgb(P_proj_vis[start_at:start_at+n_patches]), figsize=figsize, 
                  ncol=ncol, title=title, scale_each=True)
    if title:
        plt.title(title)
    plt.show()


def show_neighbors(patches, temp, img_id=0, patch_id=0,
                   title="", n_neighbors=200, ncol=20, figsize=(10, 10)):
    """Given a patch, show its neighbors on manifold.

    Args:
        patches: torch.Tensor, shape (batch_size, c, patch_size, patch_size, num_patches_per_img)
        temp: torch.Tensor, shape (batch_size, num_dim, RG**2)
        img_id: int, 0 to patches.shape[0]-1
        patch_id: int, 0 to num_patches_per_img
    
    Example call: (imgs_test defined earlier in zeyu notebook)
        batch_size=1000
        patches = unfold_image(imgs_test[:batch_size],
                            PATCH_SIZE=PATCH_SIZE, hop_length=hop_length)
        utils.show_neighbors(patches.cpu(), temp.cpu(), img_id=0, patch_id=15)
    """
    batch_size = patches.shape[0]
    num_patches = temp.shape[-1]
    num_dim = temp.shape[1]
    patch_size = patches.shape[2]
    X = rearrange(temp, "bs f n_h n_w -> (bs n_h n_w) f", bs=batch_size, n_h=num_patches, n_w=num_patches)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(X)
    one_patch = patches[img_id, :, :, :, patch_id]
    neigh_dist, neigh_id = nn.kneighbors(temp.reshape(batch_size, num_dim, num_patches**2)[img_id, :, patch_id].unsqueeze(0))
    patches_reshaped = rearrange(patches, "bs c p_h p_w n_p -> (bs n_p) c p_h p_w", bs=batch_size, c=3, p_h=patch_size, p_w=patch_size)
    sorted_patches = patches_reshaped[neigh_id]
    visualize_patches(sorted_patches[:n_neighbors], title=f"neighbors of img{img_id} patch{patch_id}", ncol=ncol)
    plt.show()


def visualize_patches(patches, title="", figsize=None, colorbar=False, 
                      ncol=None, scale_each=False):
    """
    Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches : Tensor of (batch_size, c, h, w).
    title : String; title of figure. Optional.
    ncol: Number of columns in the grid. Optional.
    figsize: Figure size. Optional.
    colorbar: Whether to display colorbar. Optional.
    scale_each: Whether to scale each image in the grid separately. Optional.
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
    out = make_grid(img_grid, padding=1, nrow=ncol, pad_value=torch.min(patches), 
        scale_each=scale_each)
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    fig = plt.gcf()
    if figsize:
        fig.set_size_inches(figsize)
    plt.imshow(out.permute(1, 2, 0))  # for images
    if colorbar:
        plt.colorbar()


def normalize_batch_rgb(img):
    """Normalize all patches to be between 0 and 1 for visualization only
    img: rgb img of dim [num_patches, c=3, ph, pw]
    axis=(2,3) because all patches are from different images so normalize each 
    patch separately
    """
    img = img - torch.amin(img, axis=(2, 3), keepdim=True)
    img = img / torch.amax(img, axis=(2, 3), keepdim=True)
    return img

