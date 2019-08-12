#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:50:09 2019

@author: ym
"""

import os
import numpy as np
from config_training import config


from scipy.io import loadmat
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
import sys
from step1 import step1_python
import warnings


def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
        
        
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def savenpy(id,annos,filelist,data_path,prep_folder):        
    resolution = np.array([1,1,1])
    name = filelist[id]
    shortname = name[:-7]
    label = annos[annos[:,0]==shortname]
    label = label[:,[3,1,2,4]].astype('float')
    
    im, m1, m2, spacing, origin = step1_python(os.path.join(data_path,name))
    Mask = m1+m2
    
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')

    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask.astype(np.int) - Mask.astype(np.int)
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    
    #save image into nii.gz to check
#    import nibabel as nib
#    new_image = nib.Nifti1Image(np.squeeze(sliceim), affine=np.eye(4))
#    nib.save(new_image, os.path.join('./','test4d.nii.gz'))
    
    np.save(os.path.join(prep_folder,shortname+'_clean.npy'),sliceim)
    np.save(os.path.join(prep_folder,shortname+'_spacing.npy'), spacing)
    np.save(os.path.join(prep_folder,shortname+'_extendbox.npy'), extendbox)
    np.save(os.path.join(prep_folder,shortname+'_origin.npy'), origin)
    np.save(os.path.join(prep_folder,shortname+'_mask.npy'), Mask)

    
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    else:
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,shortname+'_label.npy'),label2)
    print(name)
    
    
def full_prep():
    warnings.filterwarnings("ignore")

    prep_folder = "/data/LunaProj/LIDC/processed/"
    data_path = "/data/LunaProj/LIDC/merged/"
    
    alllabelfiles = "/LungNodule_DL/LIDC/labels/new_nodule.csv" #this is the file contains all nodule locations for LIDC
    
    alllabel = np.array(pandas.read_csv(alllabelfiles))
    filelist = os.listdir(data_path)

    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    print('starting preprocessing')
    pool = Pool()
    partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

    N = len(filelist)
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    
if __name__=='__main__':
    full_prep()
