# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:15:03 2019

@author: Raul Alfredo de Sousa Silva

Observation: Data used in this project is a part of the BRATS dataset used in 
the Decathlon Medical Segmentation and can be found at:
    https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
    Please read the instructions to use this code.    
    
"""
#%% 
'''
 INSTRUCTIONS:
     You can easily generate the filenames which you will manipulate by running 
     this part of the code with the path to the images on your computer completed 
     in the variable datapath below.
'''

# Data file names

datapath = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/imagesTr/"
f=open('datafnames.txt','w')
for number in range(1,51):
    datafname = datapath + "BRATS_"+ format(number,"03d") + ".nii.gz"
    f.write(datafname)
    f.write('\n')
f.close()


# Labels file names

datapath = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/labelsTr/"
f=open('labelfnames.txt','w')
for number in range(1,51):
    labelfname = datapath + "BRATS_"+ format(number,"03d") + ".nii.gz"
    f.write(labelfname)
    f.write('\n')
f.close()

# Results file names

datapath = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/resultsTr/"
f=open('resultfnames.txt','w')
for number in range(1,51):
    resultfname = datapath + "BRATS_"+ format(number,"03d") + ".nii.gz"
    f.write(resultfname)
    f.write('\n')
f.close()

#%%

# Computing paramaters
'''
INSTRUCTIONS:
    Code: computing automatically parameters as mean, variance and enclosing 
    box over the two regions with a simple indication from user of what are the
    regions composing the healthy brain and the edema.
    Results are saved in the parameters.csv file
    The second part of this script read this '.csv' file and apply the segmentation.
    
    Images will be presented one at a time to you. What you are expected to do
    is to define a bounding box around a part of the image representative of 
    the brain (class 0) and the edema/tumor (class 1) by clicking with the left
    button in the image in the top left corner and bottom right corner of the
    bounding box.
    
    At the end you must click with the right button of the mouse to validate 
    your choices. You can simply continue with left clicks if you want to 
    redefine your regions (each 4 clicks are considered as the top and bottom 
    corners of each class), only the last 4 will be considered.
    
    If the image presented to you doesn't seem to have any edema/tumor, or 
    doesn't seem to be a good example, you can simply click with center button 
    of your mouse and type in the console the number of the slice for which you
    would like to go (default is 90), by this you can look for the slice of the
    image that seems to fit the best to estimate the parameters to apply the 
    algorithm.
'''

import nilearn as nl                 # To charge images
import matplotlib.pyplot as plt      # To plot graphics
import numpy as np                   # For general manipulation
import pandas as pd                  # Create and manipulate files with important data

mu_0 = []
mu_1 = []
var0 = []
var1 = []
c00 = []
c01 = []
c02 = []
c03 = []
c10 = []
c11 = []
c12 = []
c13 = []
p = []
def onclick(event):
    x = event.xdata
    y = event.ydata
    global but
    but = event.button
    global coords
    coords = [x,y]
    return coords,but

def choice_region(image,pos):
    c = np.zeros(8).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(image[:,:,pos],cmap ='gray')
    ax.axis("off")    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    i=0
    while(but != 3 and but != 2):
        if(i%4==0):
            print("Chose the top of the box representing region 0")
            test=plt.waitforbuttonpress()
            if (but==1):
                c[0] = int(coords[0])
                c[1] = int(coords[1])
        elif(i%4==1):
            print("Chose the bottom of the box representing region 0")
            test=plt.waitforbuttonpress()
            if (but==1):
                c[2] = int(coords[0])
                c[3] = int(coords[1])
        elif(i%4==2):
            print("Chose the top of the box representing region 1")
            test=plt.waitforbuttonpress()
            if (but==1):
                c[4] = int(coords[0])
                c[5] = int(coords[1])
        else:
            print("Chose the bottom of the box representing region 1")
            test=plt.waitforbuttonpress()
            if (but==1):
                c[6] = int(coords[0])
                c[7] = int(coords[1])
        i+=1
    
    print("Done!!!")
    fig.canvas.mpl_disconnect(cid)
    plt.close('all')
    
    
    return c

# Choose one of the two approaches above to manipulate files

# 1 - Doing by list of filenames
    
f1=open('datafnames.txt','r')
number = 0
#fn = f1.readlines()[:10]
for datafname in f1:
    number += 1
    datafname = datafname[:-1]
# 2 - Doing by numbering
#for number in range(1,51):  
#    datafname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/results_tr_comb/result_"+ format(number,"03d") + ".nii.gz"
    
    print("Treating image ",number)
    c = np.zeros(8).astype(np.uint8)
    
    data = nl.image.load_img(datafname)
    image = data.get_data() 

    # Preprocessing: Taking a cut of the image
    # Function to take positions
    mu_0n  = 0
    var_0n = 0
    mu_1n  = 0
    var_1n = 0 
    but = 0
    image = image[:,:,:,0]
    pos = 90
    while (but !=3):
        coords = []
        but = 0
        c = choice_region(image,pos)
        if (but == 2):
            print("It seems that layer {0} didn't fit fo you".format(pos))
            pos = int(input("For which layer would you like to go ? [0-155]"))
            continue
        
        # Chosing regions to take parametersn
        u = np.sort(np.reshape(image[c[1]:c[3],c[0]:c[2],pos],-1))
        mu_0n  = np.mean(u[int(3*len(u)/4):-1])
        var_0n =  np.var(u[int(3*len(u)/4):-1])
        u = np.sort(np.reshape(image[c[5]:c[7],c[4]:c[6],pos],-1))
        mu_1n  = np.mean(u[0:int(len(u)/4)])
        var_1n =  np.var(u[0:int(len(u)/4)])
        
    del data, image           
    mu_0.append(mu_0n)
    mu_1.append(mu_1n)
    var0.append(var_0n)
    var1.append(var_1n)
    c00.append(c[0])
    c01.append(c[1])
    c02.append(c[2])
    c03.append(c[3])
    c10.append(c[4])
    c11.append(c[5])
    c12.append(c[6])
    c13.append(c[7])
    p.append(pos)

parameters = pd.DataFrame({'mu_0':mu_0, 'mu_1':mu_1, 'sigma0':var0, 'sigma1':var1, 'c00':c00, 'c01':c01, 'c02':c02, 'c03':c03, 'c10':c10, 'c11':c11, 'c12':c12, 'c13':c13, 'pos':p})
parameters.to_csv('parameters.csv')
f1.close()

#%% Computing minimal surfaces
'''
INSTRUCTIONS:
    This part of the script takes the '.csv' file generated from the first part 
    and generates the segmentation of the edema, which is saved in the file
    indicated by the variable resultfname.
    The code is made in a manner that you can't load a '.txt' file with the address
    of the file (image) you want to manipulate. This can easily automatize the
    task.
    Choose between one of the three approaches presented in the work that are
    implemented in the ms file:
    Approach 1: ms.segmentation_brain(datafname,mu_0,mu_1,sigma,resultfname)
    Approach 2: ms.segmentation_brain_inf(datafname,c0,c1,p,resultfname)
    Approach 3: ms.segmentation_brain_comb(datafname, c0, c1, p, mu_0, mu_1, 
                                           sigma = 1e9, resultfname)
'''
# Extracting actual data 
import ms                # Library with the main algorithm
import pandas as pd      # Create and manipulate files with important data
import numpy as np       # For general manipulation

times = []
parameters = pd.read_csv('parameters.csv',index_col = 0)

# Choose one of the two approaches above to manipulate files

# 1 - Doing by list of filenames

f1=open('datafnames.txt','r')
f2=open('resultfnames.txt','r')
number = 0
#fn = f1.readlines()[:10]
for datafname, resultfname in zip(f1,f2):
    number += 1
    datafname   = datafname[:-1]
    resultfname = resultfname[:-1]

# 2 - Doing by numbering
#for number in range(1,51):
#    number = 1
#    datafname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/imagesTr/BRATS_"+ format(number,"03d") + ".nii.gz"
#    resultfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/resultsTr/result_"+ format(number,"03d") + ".nii.gz"
    
    print("Treating image {0}".format(number))
    
    mu_0 = parameters.mu_0[number-1]
    mu_1 = parameters.mu_1[number-1]
    sigma = min(parameters.sigma0[number-1], parameters.sigma1[number-1])
    c0 = parameters.loc[number-1,'c00':'c03'].astype(np.uint8)
    c1 = parameters.loc[number-1,'c10':'c13'].astype(np.uint8)
    p = parameters.loc[number-1,'pos'].astype(np.uint8)
    if sigma != 0:
#        resultfname,tm = ms.segmentation_brain(datafname,mu_0,mu_1,sigma,resultfname)
        resultfname,tm = ms.segmentation_brain_inf(datafname,c0,c1,p,resultfname)
#       resultfname,tm = ms.segmentation_brain_comb(datafname,c0,c1,p,mu_0,mu_1,sigma = 1e9, resultfname)
    times.append(tm)

texecution = pd.DataFrame({'time': times})
texecution.to_csv('times.csv')
f1.close()
f2.close()
times = pd.read_csv('times.csv',index_col = 0)     
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
import numpy as np                   # For general manipulation
import metrics as m                  # Functions to compute metrics
import pandas as pd                  # Create and manipulate files with important data

dice = []
vd = []
sd = []
p = []
s = []

# Choose one of the two approaches above to manipulate files

# 1 - Doing by list of filenames

f1=open('labelfnames.txt','r')
f2=open('resultfnames.txt','r')
number = 0

for labelfname, resultfname in zip(f1,f2):
    # Eliminate \n
    number += 1 
    labelfname=labelfname[:-1]
    resultfname=resultfname[:-1]
    
    
# 2 - Doing by numbering
    
#for number in range(1,51):
    number = 47
    labelfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/labelsTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    resultfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/results_tr_comb/result_"+ format(number,"03d") + ".nii.gz"
    
    # Loading files
    label =  nl.image.load_img(labelfname)
    result = nl.image.load_img(resultfname)
    seg = result.get_data()
    ref = label.get_data()
    del label, result

#######################################################################
    ref = (ref>0).astype(np.uint8)
    seg = (seg>0).astype(np.uint8)
#######################################################################
    met = m.compute_metrics(seg,ref)
    print("Metrics for image {0}".format(number))
    print("DICE: {0}, VD: {1}, SD_avg: {2}".format(met[0],met[1],met[2]))
    dice.append(met[0])
    vd.append(met[1])
    sd.append(met[2])
    p.append(met[3])
    s.append(met[4])

metrics_comb_8 = pd.DataFrame({'dice':dice, 'voldif':vd, 'sumdif':sd,
                               'precision':p, 'sensitivity':s})
metrics_comb_8.to_csv('metrics_comb_8.csv')
f1.close()
f2.close()

#%%

'''
INSTRUCTIONS:
This final part of this script is just an easy way to generate images just like 
those presented on the report:
    The original image
    The proposed segmentation
    The actual segmentation
    The superposition of the two images (green to ref, red to proposed)
    The contour of the two segmentation over the original image.
    (green to ref, red to proposed)
'''

import nilearn as nl                 # To charge images
import matplotlib.pyplot as plt      # To plot graphics
import numpy as np
import metrics as m

plt.close("all")

# Choose one of the two approaches above to manipulate files

# 1 - Doing by list of filenames

f1=open('datafnames.txt','r')
f2=open('labelfnames.txt','r')
f3=open('resultfnames_comb.txt','r')
number = 0

for datafname, labelfname, resultfname in zip(f1,f2,f3):
    
    number +=1
    # Eliminate \n
    datafname=datafname[:-1]
    labelfname=labelfname[:-1]
    resultfname=resultfname[:-1]
    
# 2 - Doing by numbering
    
#for number in range(1,51):
    number = 7
    datafname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/imagesTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    labelfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/labelsTr/BRATS_"+ format(number,"03d") + ".nii.gz"
    resultfname = "C:/Users/raul-/Documents/PRAT_projet/Task01_BrainTumour/resultsTr/result_"+ format(number,"03d") + ".nii.gz"
    
    # Loading files
    data = nl.image.load_img(datafname)
    label =  nl.image.load_img(labelfname)
    result = nl.image.load_img(resultfname)
    seg = result.get_data()
    image = data.get_data() 
    ref = label.get_data()
    del data, label, result
    
    # Choosing cut
    pos = 90

    m.draw_contour(seg[:,:,pos],ref[:,:,pos],image[:,:,pos,0],number,pos)
    # Postprocessing: Taking a cut of the image
    plt.figure(1)
    plt.imshow(image[:,:,pos,0],cmap ='gray')
    plt.axis("off")
    plt.savefig('images/image_{0}cut_{1}.png'.format(number,pos))
    
    # Postprocessing: Obtained segmentation
    plt.figure(2)
    plt.imshow(seg[:,:,pos],cmap ='gray')
    plt.axis("off")
    plt.savefig('images/segmentation_{0}cut_{1}.png'.format(number,pos))
    
    # Postprocessing: Refecence segmentattion
    plt.figure(3)
    plt.imshow((ref[:,:,pos]),cmap ='gray')
    plt.axis("off")
    plt.savefig('images/reference_{0}cut_{1}.png'.format(number,pos))
    
    # Postprocessing: comparing segmentations
    conf = np.zeros([image.shape[0],image.shape[1],3])
    conf[:,:,0] = 255*(seg[:,:,pos])
    conf[:,:,1] = ref[:,:,pos]
    
    plt.figure(4)
    plt.imshow(conf)
    plt.axis("off")
    plt.savefig('images/comparison_{0}cut_{1}.png'.format(number,pos))
    
    m.draw_contour(seg[:,:,pos],ref[:,:,pos],image[:,:,pos,0],number,pos)

f1.close()
f2.close()
f3.close()

#%%

