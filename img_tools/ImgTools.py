# imports
import numpy as np
from ipywidgets import *
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from torchvision.utils import make_grid

"""
Function: imshow
    
    Function used to display a given Tensor defined image
    
        Args:
            img (torch.Tensor): image to be displayed
            ax  (plt.subplots axis) : subplot axis
"""


def imshow(img, ax=None):

    if not isinstance(img, np.ndarray):
        # img -> CPU for numpy
        img = img.cpu()
        # (C, H. W) -> (H, W, C)
        np_img = np.transpose(
            img.numpy(),
            (1, 2, 0)
        )
    else:
        np_img = img
    
    if ax is None:
        # plt img
        plt.figure(figsize=(10, 5))
        plt.imshow(np_img)
        
    else:
        ax.imshow(np_img)
    
    return


"""
Function: save_img
    
    Function used to save a given Tensor defined image
    
        Args:
            img (torch.Tensor) : image to be displayed
            save_loc (string)  : image save location
"""


def save_img(img, save_loc='./img.png'):
    
    # img -> CPU for numpy
    img = img.cpu()

    # (C, H. W) -> (H, W, C)
    np_img = np.transpose(
        img.numpy(),
        (1, 2, 0)
    )*255
    
    # PIL image
    pil_img = PIL_Image.fromarray(np_img.astype('uint8'))
    
    # save image losslessly
    pil_img.save(save_loc, 'PNG')
    
    
    return


"""
Function: display_patches

    Function used to display a group of patches
        
        Args: 
            p (torch.Tensor): Tensor containing patch array
            n (int): patches to place per row
"""


def disp_patches(patches, nrow, pad=1, pad_val=1, norm=False):
    
    # plot image patches
   
    p = make_grid(
        tensor=patches,
        nrow=nrow,
        padding=pad,
        normalize=norm,
        pad_value=pad_val
    )
    
    # display patches
    imshow(p)
    
    return


"""
Function: patches2img
    
    Function used to stich together patches into a single Tensor image
        
        Args:
            pacthes (torch.Tensor) : group of patches belonging to a single image
            nr      (int)          : number of patches making up a row of the image
       Return:
            img_t   (torch.Tensor) : reconstructed Tensor image
"""


def patches2img(patches, nr, pad=0):
    
    # create img tensor from patches
    
    if len(patches.size()) == 5:
        # (b,t,c,h,w) --> (t,c,h,w)
        patches = patches[0]
    
    # put on cpu
    patches = patches.cpu()
    
    img_t = make_grid(
        patches, 
        nrow=nr,
        padding=pad
    )
    
    return img_t


"""
Function: disp_images_widget
    
    Function used to display a group of pictures as a widget
    
    Args:
        images (torch.Tensor) : group of images
"""


def disp_images_widget(title, imgs, psnr, ssim, bpp):

    n = imgs.size(0)

    interact(
        imshow_callback,
        imgs=fixed(imgs),
        bpp=fixed(bpp),
        ssim=fixed(ssim),
        psnr=fixed(psnr),
        title=fixed(title),
        i=(1, n, 1)
    )

    return


def imshow_callback(imgs, title, psnr, ssim, bpp, i):

    # disp img
    imshow(imgs[i-1])

    setup_plot(
        title=title,
        y_label="Bpp : " + str(bpp[i-1]),
        x_label="PSNR : " + str(psnr[i-1]) + "\nSSIM : " + str(ssim[i-1]),
        newfig=False
    )
    
    return


"""
Function: vs_imshow
    
    display refernce image next to compressed image
    
    Args:
        ref  (torch.Tensor) : reference image
        comp (torch.Tensor) : compressed image
        pad  (int)          : padding btwn images
        
"""


def vs_imshow(ref, comp, pad=2):
    
    # display ref nxt2 comp image   
    img = make_grid([ref.cpu(), comp.cpu()], nrow=2)
    imshow(img)
    
    return


"""
Function: setup_plot
    
    creates a new labled plot
    
    ref: Systems & Signals 414 Coures Practical Template
"""


def setup_plot(title, y_label='', x_label='', newfig=True):
    
    if newfig:
        plt.figure()
        
    plt.title(title, fontdict={'fontsize':'xx-large'})
    
    # x & y axes labels
    plt.ylabel(y_label, fontsize='x-large')
    plt.xlabel(x_label, fontsize='x-large')
    
    return


"""
Function: setup_subplot
    
    creates a new labled subplot
    
    ref: https://matplotlib.org/1.4.1/users/tight_layout_guide.html
    
"""


def setup_subplot(ax, title, y_label='', x_label=''):
    
    ax.set_xlabel(x_label, fontsize='x-large')
    ax.set_ylabel(y_label, fontsize='x-large')
    ax.set_title(title, fontsize='x-large')
    
    return


"""
    disp_prog_imgs
    
        display a group of pictures as well as their bpp, metric values as a series pf sub_plots
    
    ref : https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
"""


def disp_prog_imgs(title, imgs, bpp, ssim, psnr, start_itr=0):

    # set up grid
    fig, axarr = plt.subplots(1, len(bpp), constrained_layout=True, figsize=(10, 5))

    for i, img in enumerate(imgs):

        # set up subplot
        setup_subplot(
            axarr[i],
            title="Iteration : " + str(start_itr+i+1),
            x_label="PSNR : "+str(psnr[i])+"\nSSIM : "+str(ssim[i]),
            y_label="Bpp : "+str(bpp[i])
        )

        # disp img
        imshow(img, axarr[i])

    fig.suptitle(title, fontsize=20, y=0.9)

    # uncomment to save figure
    #fig.savefig(title + ".pdf")

    return


"""
Function display_inpaint:
    
    use to display inpainting results
    
        Args:
            context (torch.Tensor)      :  context region
            inpaint (torch.Tensor)      :  inpainting prediction
            ground_truth (torch.Tensor) :  ground truth patch
"""


def display_inpaint(context, inpaint, ground_truth):

    # set up grid
    fig, axarr = plt.subplots(1, 3, figsize=(20, 8))

    # plot titles
    fig.suptitle("Inpainting Test", fontsize=20)

    setup_subplot(axarr[0], title="Context Region")
    imshow(context, axarr[0])

    setup_subplot(axarr[1], title="Inpainting")
    imshow(inpaint, axarr[1])

    setup_subplot(axarr[2], title="Ground Truth")
    imshow(ground_truth, axarr[2])

    return
