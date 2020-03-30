# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:11:07 2019
@author: Raul Alfredo de Sousa Silva

Description:
    This code is designed to reproduce the work of the PhD thesis of Jean Stawiaski (reference above) especially
    concerning the segmentation of medical images with a minimal surface over a region adjacency graph.
    In a nutshell what is made here is:
        - Charging images from database
        - Computing morphological gradient over a 3D image (We chose to work firstly with the FLAIR image)
        - Computing the low level watershed of the image
        - Creating the region adjacency graph (computing weights, and affecting them to respective nodes)
        - Computing the Grathcut
        - Plotting results
        - Saving segmentation
    
    What we must do in next steps:
        - Provide ability to insert markers (scribbles, as in Stawiaski's work)
        - To be able to separate into all 4 classes the tumor
    
Reference:
    Stawiaski, J. Mathematical morphology and graphs: Application to interactive medical image segmentation
    (Doctoral dissertation). Ecole Nationale Superieure des Mines de Paris, 2008
"""

# Charging libraries

# Nilearn: VERSION: 0.5.2
# Matplotlib: VERSION: 3.1.1
# Numpy: VERSION: 1.16.5
# Skimage: VERSION: 0.15.0
# Scipy: VERSION: 1.3.1
# Maxflow: VERSION: 1.2.12
# Nibabel: VERSION: 2.5.1

from nilearn import image as img                # To charge images
import numpy as np                   # For general manipulation
import skimage.morphology as skm     # Many morphological operations including watershed
from skimage.feature import peak_local_max        # Some complementary functions to labelise regions
from scipy.ndimage import label     # Complementary function to labelaize regions
import maxflow                       # Computing minimal surface with a graphcut approach 
from nibabel import Nifti1Image                # Create output file with the result
import time                          # Measure processing time
import os

def segmentation_brain(datafname,mu_0,mu_1,var=1,resultfname="/results_tr/result_XXX.ni.gz"):
    
    tic = time.clock()
    # Part 1 - Preprocessing (charging file computing watershed)    
    # Extracting actual data
    data = img.load_img(datafname)
    image = data.get_data() 
    
    # Computing morphological gradient over a 3D image
    flair = image[:,:,:,0]
    strel = skm.ball(1)
    grad= skm.dilation(flair,strel)-skm.erosion(flair,strel)
    # Computing the low level watershed of the image
    local_mini = peak_local_max(grad.max()-grad, indices=False)
    markers = label(local_mini)[0]
    labels = skm.watershed(grad, markers, watershed_line=False) # To pick regions
    
    M = labels.max()
    print("")
    print("Regions founded: ",M)
    
    # Part 2 - Creating the region adjacency graph (computing weights, and affecting them to respective nodes)
    print("Creating the graph...")
    nodesmax = M                                # One node for each region and for each possible label 
    edgesmax = nodesmax*10                         # Every node has the mean number of neighbors
    G = maxflow.Graph[float](nodesmax,edgesmax)
    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
        ind = np.nonzero((labels == i+1))
        mu = np.mean(image[ind])
        tw = (mu-mu_0)**2/(2*var)
        sw = (mu-mu_1)**2/(2*var)
        G.add_nodes(1)
        G.add_tedge(i,sw,tw)
        
        if (i==0):                   # There's nothing else to do (points in 1 are useless and computationally costful)
            continue
        
        indx,indy,indz = np.nonzero((labels == i+1))
        ind = []
        weight = []
        for n in range(len(indx)):
            point = (indx[n],indy[n],indz[n])
            neighs = [(indx[n]+1,indy[n],indz[n])]
            neighs.append((indx[n]-1,indy[n],indz[n]))
            neighs.append((indx[n],indy[n]+1,indz[n]))
            neighs.append((indx[n],indy[n]-1,indz[n]))
            neighs.append((indx[n],indy[n],indz[n]+1))
            neighs.append((indx[n],indy[n],indz[n]-1))
            for neigh in neighs:
                j = labels[neigh]-1
                if (j < i):
                    if (j,i) not in ind:
                        ind.append((j,i))
                        weight.append( (1+max(grad[point],grad[neigh]))**(-2) )                        
                    else:
                        index = ind.index((j,i))
                        weight[index]+= (1+max(grad[point],grad[neigh]))**(-2)
        for w in range(len(ind)):
            p,q = ind[w]
            G.add_edge(p,q,weight[w],weight[w])
    print("Progress:  100 %")
    
    # Part 3 - Computing the Graph-cut
    print("Finding best cut...")
    cost = G.maxflow()
    print("Cost of this cut: ",cost)
    print("Recomposing image...")
    
    seg = np.zeros(image.shape[0:3])
    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
        indices = np.nonzero(labels == i+1)
        seg[indices] = G.get_segment(i)
    print("Progress: 100 %")
    
    toc = time.clock()
    tm = toc - tic
    print("Processing time to one image: ",tm)
    print("Image segmented! Manipulate the variable 'seg' to see the results.")
    
    # Saving segmentation obtained
    t = resultfname.rfind("/")
    try:
        os.makedirs(resultfname[0:t+1])
    except:
        print("Folder already exists")
    else:
        print("Folder created")

    result = Nifti1Image(seg, affine=np.eye(4)) 
    result.to_filename(resultfname)
    
    return resultfname,tm

def segmentation_brain_inf(datafname,c0,c1,p,resultfname="/results_tr/result_XXX.ni.gz"):
    
    tic = time.clock()
    # Part 1 - Preprocessing (charging file computing watershed)    
    data = img.load_img(datafname)
    image = data.get_data() 
    
    # Computing morphological gradient over a 3D image
    flair = image[:,:,:,0]
    strel = skm.ball(1)
    grad= skm.dilation(flair,strel)-skm.erosion(flair,strel)
    # Computing the low level watershed of the image
    local_mini = peak_local_max(grad.max()-grad, indices=False)
    markers = label(local_mini)[0]
    labels = skm.watershed(grad, markers, watershed_line=False) # To pick regions
    
    M = labels.max()
    print("")
    print("Regions founded: ",M)
    
    # Part 2 - Creating the region adjacency graph (computing weights, and affecting them to respective nodes)
    print("Creating the graph...")
    nodesmax = M                                # One node for each region and for each possible label 
    edgesmax = nodesmax*10                         # Every node has the mean number of neighbors
    G = maxflow.Graph[float](nodesmax,edgesmax)
    
    lab0 = np.unique(labels[c0[1]:c0[3],c0[0]:c0[2],p])
    lab1 = np.unique(labels[c1[1]:c1[3],c1[0]:c1[2],p])

    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
            
        G.add_nodes(1)
        if (i==0):                   # There's nothing else to do (points in 1 are useless and computationally costful)
            continue

        if(i+1 in lab1):
            G.add_tedge(i,0,1000)
        if(i+1 in lab0):
            G.add_tedge(i,1000,0)
        
        indx,indy,indz = np.nonzero((labels == i+1))
        ind = []
        weight = []
        for n in range(len(indx)):
            point = (indx[n],indy[n],indz[n])
            neighs = [(indx[n]+1,indy[n],indz[n])]
            neighs.append((indx[n]-1,indy[n],indz[n]))
            neighs.append((indx[n],indy[n]+1,indz[n]))
            neighs.append((indx[n],indy[n]-1,indz[n]))
            neighs.append((indx[n],indy[n],indz[n]+1))
            neighs.append((indx[n],indy[n],indz[n]-1))
            for neigh in neighs:
                j = labels[neigh]-1
                if (j < i):
                    if (j,i) not in ind:
                        ind.append((j,i))
                        weight.append( (1+max(grad[point],grad[neigh]))**(-2) )                        
                    else:
                        index = ind.index((j,i))
                        weight[index]+= (1+max(grad[point],grad[neigh]))**(-2)
        for w in range(len(ind)):
            p,q = ind[w]
            G.add_edge(p,q,weight[w],weight[w])
    print("Progress:  100 %")
    
    # Part 3 - Computing the Graph-cut
    print("Finding best cut...")
    cost = G.maxflow()
    print("Cost of this cut: ",cost)
    
    print("Recomposing image...")
    
    seg = np.zeros(image.shape[0:3])
    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
        indices = np.nonzero(labels == i+1)
        seg[indices] = G.get_segment(i)
    print("Progress: 100 %")
    
    toc = time.clock()
    tm = toc-tic
    print("Processing time to one image: ",tm)
    print("Image segmented! Manipulate the variable 'seg' to see the results.")
    
    # Saving segmentation obtained
    t = resultfname.rfind("/")
    try:
        os.makedirs(resultfname[0:t+1])
    except:
        print("Folder already exists")
    else:
        print("Folder created")

    result = Nifti1Image(seg, affine=np.eye(4)) # It's compulsory the passage of a affine matrix as an argument
    result.to_filename(resultfname)
    
    return resultfname,tm

def segmentation_brain_comb(datafname,c0,c1,p,mu_0,mu_1,var=1e9,resultfname="/results_tr/result_XXX.ni.gz"):
    
    tic = time.clock()
    
    # Part 1 - Preprocessing (charging file computing watershed)    
    data = img.load_img(datafname)
    image = data.get_data() 
    
    # Computing morphological gradient over a 3D image
    flair = image[:,:,:,0]
    strel = skm.ball(1)
    grad= skm.dilation(flair,strel)-skm.erosion(flair,strel)
    # Computing the low level watershed of the image
    local_mini = peak_local_max(grad.max()-grad, indices=False)
    markers = label(local_mini)[0]
    labels = skm.watershed(grad, markers, watershed_line=False) # To pick regions
    
    M = labels.max()
    print("")
    print("Regions founded: ",M)
    
    # Part 2 - Creating the region adjacency graph (computing weights, and affecting them to respective nodes)
    print("Creating the graph...")
    nodesmax = M                                # One node for each region and for each possible label 
    edgesmax = nodesmax*10                         # Every node has the mean number of neighbors
    G = maxflow.Graph[float](nodesmax,edgesmax)
    G.add_nodes(M)
    
    lab0 = np.unique(labels[c0[1]:c0[3],c0[0]:c0[2],p])
    lab1 = np.unique(labels[c1[1]:c1[3],c1[0]:c1[2],p])
    if 1 in lab1:
        lab1 = lab1[1:-1]

    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
    
        if (i==0):                   # There's nothing else to do (points in 1 are useless and computationally costful)
            continue
        
        if(i+1 in lab1):
            G.add_tedge(i,0,1000)
        elif(i+1 in lab0):
            G.add_tedge(i,1000,0)
        else:
            ind = np.nonzero((labels == i+1))
            mu = np.mean(image[ind])
            tw = (mu-mu_0)**2/(2*var)
            sw = (mu-mu_1)**2/(2*var)
            G.add_tedge(i,sw,tw)
        
        indx,indy,indz = np.nonzero((labels == i+1))
        ind = []
        weight = []
        for n in range(len(indx)):
            point = (indx[n],indy[n],indz[n])
            neighs = [(indx[n]+1,indy[n],indz[n])]
            neighs.append((indx[n]-1,indy[n],indz[n]))
            neighs.append((indx[n],indy[n]+1,indz[n]))
            neighs.append((indx[n],indy[n]-1,indz[n]))
            neighs.append((indx[n],indy[n],indz[n]+1))
            neighs.append((indx[n],indy[n],indz[n]-1))
            for neigh in neighs:
                j = labels[neigh]-1
                if (j < i):
                    if (j,i) not in ind:
                        ind.append((j,i))
                        weight.append( (1+max(grad[point],grad[neigh]))**(-2) )                        
                    else:
                        index = ind.index((j,i))
                        weight[index]+= (1+max(grad[point],grad[neigh]))**(-2)
        for w in range(len(ind)):
            p,q = ind[w]
            G.add_edge(p,q,weight[w],weight[w])
    print("Progress:  100 %")
    
    # Part 3 - Computing the Graph-cut
    print("Finding best cut...")
    cost = G.maxflow()
    print("Cost of this cut: ",cost)
    print("Recomposing image...")
    
    seg = np.zeros(image.shape[0:3])
    for i in range(M):
        if (round((i/M)+.05,1) < round((i+1)/M+.05,1)):
            print("Progress: ",100*round((i/M),2),"%")
        indices = np.nonzero(labels == i+1)
        seg[indices] = G.get_segment(i)
    print("Progress: 100 %")
    
    toc = time.clock()
    tm = toc - tic
    print("Processing time to one image: ",tm)
    print("Image segmented! Manipulate the variable 'seg' to see the results.")
    
    # Saving segmentation obtained
    t = resultfname.rfind("/")
    try:
        os.makedirs(resultfname[0:t+1])
    except:
        print("Folder already exists")
    else:
        print("Folder created")
    
    result = Nifti1Image(seg, affine=np.eye(4)) 
    result.to_filename(resultfname)
    
    return resultfname,tm
