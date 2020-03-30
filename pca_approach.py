# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:38:32 2019

@author: Raul Alfredo de Sousa Silva
"""

'''
This is a complementary code to the work developed on the project about 
mathematical morphology on graphs. Actually, to solve the problem proposed, it
semms more reasonable an approach by Principal component analysis (PCA), since
we have 4 images of complementary MRI techniques.
So we try here to apply PCA to see if improvements are significative over the
technique proposed. Also, we foresee the possibility of making a multiclass 
labeling easily with PCA, so we tried to implement it here.
TO use it the only changes one should make is to change the directory name to 
the one where are the data files and to where he/she would lie to put the 
result file (datafname and resultfname, respectively) and eventually expand the
number of images to treat (now in the range 1-51).
'''

import numpy as np                   # For general manipulation
import nilearn as nl                 # To charge images
import matplotlib.pyplot as plt      # To plot graphics
from nibabel import Nifti1Image      # Create output file with the result
import skimage.morphology as skm

plt.close("all")

for number in range(1,51):
    print("Treating image ",number)
    datafname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/imagesTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    resultfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/results_tr_pca/result_"+ format(number,"03d") + ".nii.gz"
    
    data = nl.image.load_img(datafname)
    image = data.get_data() 
    pos=90
    
    m_0 = np.mean(image[:,:,:,0])
    v_0 = np.var(image[:,:,:,0])
    m_1 = np.mean(image[:,:,:,1])
    v_1 = np.var(image[:,:,:,1])
    m_2 = np.mean(image[:,:,:,2])
    v_2 = np.var(image[:,:,:,2])
    m_3 = np.mean(image[:,:,:,3])
    v_3 = np.var(image[:,:,:,3])
    
    c = [np.reshape((image[:,:,:,0]-m_0)/v_0,-1),
         np.reshape((image[:,:,:,1]-m_1)/v_1,-1),
         np.reshape((image[:,:,:,3]-m_2)/v_2,-1),
         np.reshape((image[:,:,:,3]-m_3)/v_3,-1)]
    Xp = np.array(c)
    
    # PCA
    C = Xp@Xp.T
    e1,e2 = np.linalg.eig(C)
    prop = np.cumsum(e1)/np.sum(e1)
    y = Xp.T@e2[:,0]
    
    # Reconstructing image
    im = np.reshape(y,(240,240,155))
    im=abs(im)
    
    # Detecting class 1
    seuil = 0.022
    seg = (im>seuil).astype(np.uint8)
    seg = seg*(image[:,:,:,0]>700)
    
    # Eliminating spurious detections
    strel = skm.ball(1)
    seg = skm.opening(seg,strel)
    
    # Detecting class 3
    seg = seg + 2*(image[:,:,:,2]>1000)*seg
    
    # Detecting class 2
    
    y = Xp.T@e2[:,1]
    im = np.reshape(y,(240,240,155))
    im=abs(im)
    seuil = 0.004
    seg = seg + (seg==1)*(im>seuil).astype(np.uint8)
    
    
    
    result = Nifti1Image(seg, affine=np.eye(4)) # It's compulsory the passage of a affine matrix as an argument
    result.to_filename(resultfname)
    
    del Xp,y,image,im
    
#%%
''' 
INSTRUCTIONS:
This part of the code compute the metrics defined in the work to measure the 
quality of the segmentation.
You can't simply pass the corresponding file names of both reference and segmentation
and metrics will be computed.
It's made in such a way that the metrics are computed over all tumor.
If you would like to evaluate just specific labels such as edema or tumor, please
change lines just between the hashes.
'''

import nilearn as nl                 # To charge images
import matplotlib.pyplot as plt      # To plot graphics
import numpy as np                   # For general manipulation
import metrics as m                  # Fonctions de calcul des métriques
import pandas as pd                  # Create and manipulate files with important data

plt.close("all")

dice = []
vd = []
sd = []
p = []
s = []

for number in range(1,51):    
    
    pos = 90
    
    # Preprocessing: Chosing file
    #datafname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/imagesTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    labelfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/labelsTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    resultfname_pca = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/results_tr_pca/result_"+ format(number,"03d") + ".nii.gz"
    
    #data = nl.image.load_img(datafname)
    result_pca = nl.image.load_img(resultfname_pca)
    label =  nl.image.load_img(labelfname)
    
    
    #image = data.get_data() 
    segmentation = label.get_data()
    seg_pca = result_pca.get_data()
    
    del label, result_pca
    
#######################################################################
    ref = (segmentation>0).astype(np.uint8)
    seg = (seg_pca>0).astype(np.uint8)
#######################################################################
    met = m.compute_metrics(seg,ref)
    print("Metrics for image {0}".format(number))
    print("DICE: {0}, VD: {1}, SD_avg: {2}".format(met[0],met[1],met[2]))
    dice.append(met[0])
    vd.append(met[1])
    sd.append(met[2])
    p.append(met[3])
    s.append(met[4])
    
    metrics_pca = pd.DataFrame({'dice':dice, 'voldif':vd, 'sumdif':sd,
                                   'precision':p, 'sensitivity':s})
    metrics_pca.to_csv('metrics_pca.csv')
    
    del seg_pca,segmentation
