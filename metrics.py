# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:25:10 2019

@author: Raul Alfredo de Sousa Silva

Library to compute the metrics defined on the work:
    Dice
    Relative absolute volumetric difference
    Average symmetric absolute surface distance
    Precision
    Sensitivity
"""
import numpy as np

# Function draw_contour
def draw_contour(seg, segmentation,image,number=0,pos=0,n_im=5):
    '''
    Draw the contour over the given image.
    seg: contour image (2D) obtained by your method
    sementation: reference contour image (2D)
    image: image over which the contour will be drawn
    There's no output, image with contour is automatically plotted.
    '''
    from skimage.morphology import dilation,disk
    from matplotlib.pyplot import figure,imshow,axis,savefig
    
    strel = disk(1)
    cont = dilation(seg,strel)-seg
    u = (segmentation>0).astype(np.int8)
    cont_r = dilation(u,strel)-u
    
    imres = np.zeros((cont.shape[0],cont.shape[1],4))
    imref = np.zeros((cont.shape[0],cont.shape[1],4))
    imres[:,:,0] = cont[:,:]
    imref[:,:,1] = cont_r[:,:]
    imres[:,:,3] = cont[:,:]
    imref[:,:,3] = cont_r[:,:]
    
    figure(n_im)
    imshow(image,cmap ='gray')
    imshow(imres)
    imshow(imref)
    axis("off")
    savefig('images/contour_{0}cut_{1}.png'.format(number,pos))

def dice(seg,ref):
    return round(2*np.sum((seg>0) & (ref>0))/(np.sum(seg>0) + np.sum(ref>0)),3)

def volume_difference(seg,ref):
    return 100*round(abs(1-np.sum(seg)/np.sum(ref)),3)

def precision(seg,ref):
    return round(np.sum((seg>0) & (ref>0))/np.sum((seg>0)),3)

def sensitivity(seg,ref):
    return round(np.sum((seg>0) & (ref>0))/np.sum((ref>0)),3)

def surface_distance_abs(seg,ref):
    import skimage.morphology as skm
    if (np.sum(seg)==0):
        return 0
    se = skm.cube(3)
    seg_surface = seg - skm.erosion(seg,se)
    ref_surface = ref - skm.erosion(ref,se)
    dist = []
    ix,iy,iz = np.where(ref_surface)
    p_ref = np.array([ix,iy,iz])
    indx,indy,indz = np.where(seg_surface)
    for i in range(len(indx)):
        p = np.array([[indx[i]],[indy[i]],[indz[i]]])
        dists = np.sqrt(np.sum((np.repeat(p,p_ref.shape[1],1)-p_ref)**2,0))
        dist.append(np.min(dists))
    ix,iy,iz = np.where(seg_surface)
    p_seg = np.array([ix,iy,iz])
    indx,indy,indz = np.where(ref_surface)
    for i in range(len(indx)):
        p = np.array([[indx[i]],[indy[i]],[indz[i]]])
        dists = np.sqrt(np.sum((np.repeat(p,p_seg.shape[1],1)-p_seg)**2,0))
        dist.append(np.min(dists))
        
    return round(sum(dist)/len(dist),3)

def compute_metrics(seg,ref):
    D = dice(seg,ref)
    VD = volume_difference(seg,ref)
    SD = surface_distance_abs(seg,ref)
    P = precision(seg,ref)
    S = sensitivity(seg,ref)
    return [D,VD,SD,P,S]