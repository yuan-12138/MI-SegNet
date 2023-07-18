#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:22:39 2020

@author: yuanbi
"""

import cv2
import os
import torch
import numpy as np
from torchvision import transforms

def get_traj_index(x,length):
    row_index=np.sum(np.array(length.cumsum()) <= x, axis=0)-1
    column_index=x-length.cumsum()[row_index]
    return (row_index,column_index)
        
        
        

class Dataset_loader_test(torch.utils.data.Dataset):
    def __init__(self, list_IDs, list_length, files_img, files_label, NUM_DEMO, demo_path):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.list_IDs = list_IDs
        self.list_length=list_length
        self.files_img=files_img
        self.files_label=files_label
        self.NUM_DEMO=NUM_DEMO
        
        self.transform_image=transforms.Normalize(0.5,0.5)

        self.demo_path=demo_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        Demo_num,Frame_num=get_traj_index(ID,self.list_length)
        
        image_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'img',self.files_img[Demo_num][Frame_num])
        label_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'label',self.files_label[Demo_num][Frame_num])
        
        src = cv2.imread(image_path)
         
        img = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        
        src = cv2.imread(label_path)
        
        label = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        return img, label
        
class Dataset_loader_ft(torch.utils.data.Dataset):
    def __init__(self, list_IDs, list_length, files_img, files_label, NUM_DEMO, demo_path, train=True):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.list_IDs = list_IDs
        self.list_length=list_length
        self.files_img=files_img
        self.files_label=files_label
        self.NUM_DEMO=NUM_DEMO
        
        self.transform_image=transforms.Normalize(0.5,0.5)

        self.demo_path=demo_path
        
        self.train = train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        Demo_num,Frame_num=get_traj_index(ID,self.list_length)
        
        if self.train:
            image_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'train/img',self.files_img[Demo_num][Frame_num])
            label_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'train/label',self.files_label[Demo_num][Frame_num])
        else:
            image_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'test/img',self.files_img[Demo_num][Frame_num])
            label_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'test/label',self.files_label[Demo_num][Frame_num])
        
        src = cv2.imread(image_path)
         
        img = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        
        src = cv2.imread(label_path)
        
        label = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        return img, label
    
        
        



class Pair_Adv_Dataset_loader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, list_length, files_img, files_label, NUM_DEMO, demo_path):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.list_IDs = list_IDs
        self.list_length=list_length
        self.files_img=files_img
        self.files_label=files_label
        self.NUM_DEMO=NUM_DEMO
        
        self.transform_image=transforms.Normalize(0.5,0.5)

        self.demo_path=demo_path
        
        self.weights_aug = np.array([5, 2, 1, 4, 3])
        
        self.prob_list=[0.111,0.222,0.333,0.444,0.555,0.666,0.777,0.888,0.999]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        Demo_num,Frame_num=get_traj_index(ID,self.list_length)
        
        image_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'img',self.files_img[Demo_num][Frame_num])
        label_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'label',self.files_label[Demo_num][Frame_num])
        
        src = cv2.imread(image_path)
         
        img = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        src = cv2.imread(label_path)
        
        label = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # label = label/255
        
        aug_style_diff=0
        aug_spatial_diff=1
        while aug_style_diff<3 and aug_spatial_diff!=0:
        
            prob_style_1 = np.random.rand(5)
            alpha_style_1 = np.random.rand(5)
            prob_spatial_1 = np.random.rand(3)
            alpha_spatial_1 = np.random.rand(3)
        
            prob_style_2 = np.random.rand(5)
            alpha_style_2 = np.random.rand(5)
            prob_spatial_2 = np.random.rand(3)
            alpha_spatial_2 = np.random.rand(3)
        
            aug_style_diff=np.sum(abs((prob_style_1<0.3)*alpha_style_1*self.weights_aug-(prob_style_2<0.3)*alpha_style_2*self.weights_aug))
            
            aug_spatial_diff=np.sum((self.prob_list>prob_spatial_1[1]) * (self.prob_list<prob_spatial_1[1]+0.111)\
                * (self.prob_list>prob_spatial_2[1]) * (self.prob_list<prob_spatial_2[1]+0.111))
        
        augmented_image_1, augmented_label_1=self.data_augmentation(prob_style_1,alpha_style_1,prob_spatial_1,alpha_spatial_1,img,label)
        augmented_image_2, augmented_label_2=self.data_augmentation(prob_style_2,alpha_style_2,prob_spatial_2,alpha_spatial_2,img,label)
        augmented_image_12, _=self.data_augmentation(prob_style_1,alpha_style_1,prob_spatial_2,alpha_spatial_2,img,label)
        augmented_image_21, _=self.data_augmentation(prob_style_2,alpha_style_2,prob_spatial_1,alpha_spatial_1,img,label)
        augmented_image_1 = augmented_image_1/255
        augmented_image_2 = augmented_image_2/255
        augmented_image_12 = augmented_image_12/255
        augmented_image_21 = augmented_image_21/255

        return augmented_image_1, augmented_image_2, augmented_image_12, augmented_image_21, augmented_label_1, augmented_label_2
    
    def data_augmentation(self,prob_style,alpha_style,prob_spatial,alpha_spatial,img,label):
        if prob_spatial[0]<0.5:
            img,label=self.crop(img,label,alpha_spatial[0:2],prob_spatial[1])
        if prob_spatial[2]<0.05:
            img,label=self.flip(img,label,alpha_spatial[2])
        if prob_style[0]<0.1:
            img=self.sharpness(img,alpha_style[0])
        if prob_style[1]<0.1:
            img=self.blurriness(img,alpha_style[1])
        if prob_style[2]<0.1:
            img=self.noise_level(img,alpha_style[2])
        if prob_style[3]<0.1:
            img=self.brightness(img,alpha_style[3])
        if prob_style[4]<0.1:
            img=self.contrast(img,alpha_style[4])
        return img, label
        
    def sharpness(self,img,alpha):
        alpha=alpha*20+10 #[10,30]
        blur = cv2.GaussianBlur(img,(0,0),1.0)
        blurr = cv2.GaussianBlur(blur,(0,0),1.0)
        unsharp_image = cv2.addWeighted(blur, alpha+1, blurr, -alpha, 0)
        return unsharp_image.astype('uint8')
    
    def blurriness(self,img,alpha):
        alpha=alpha*1.25+0.25 #[0.25,1.5]
        blur_image = cv2.GaussianBlur(img,(0,0),alpha)
        return blur_image.astype('uint8')
    
    
    def noise_level(self,img,alpha):
        alpha=alpha*0.04+0.01 #[0.01,0.05]
        gaussian = np.random.normal(0, alpha, (img.shape[0],img.shape[1]))*255
        # alpha=alpha*50+30
        # rayleigh = np.random.rayleigh(1,[img.shape[0],img.shape[1]])*alpha
        noised_image=img+gaussian
        noised_image[noised_image>255]=255
        noised_image[noised_image<0]=0
        return noised_image.astype('uint8')
    
    def brightness(self,img,alpha):
        alpha=alpha*0.2-0.1
        alpha=int(alpha*255)
        brightness_image=img+alpha
        brightness_image[brightness_image>255] = 255
        brightness_image[brightness_image<0] = 0
        return brightness_image.astype('uint8')
    
    def contrast(self,img,alpha):
        alpha=alpha*2.5+0.5
        invGamma = 1 / alpha
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
    
        contrast_image=cv2.LUT(img, table)
    
        return contrast_image.astype('uint8')
    
    
    def crop(self,img,label,alpha,prob):
        alpha=alpha*0.2+0.7
        
        height, width=img.shape
    
        croped_height=int(height*alpha[0])
        croped_width=int(width*alpha[1])
    
        if prob<0.111:
            c = [0,0]
        elif prob<0.222:
            c = [0,int(width*(1-alpha[1]))]
        elif prob<0.333:
            c = [0,int(width*(1-alpha[1])//2)]
        elif prob<0.444:
            c = [int(height*(1-alpha[0])),0]
        elif prob<0.555:
            c = [int(height*(1-alpha[0])),int(width*(1-alpha[1]))]
        elif prob<0.666:
            c = [int(height*(1-alpha[0])),int(width*(1-alpha[1])//2)]
        elif prob<0.777:
            c = [int(height*(1-alpha[0])//2),0]
        elif prob<0.888:
            c = [int(height*(1-alpha[0])//2),int(width*(1-alpha[1]))]
        else:
            c = [int(height*(1-alpha[0])//2),int(width*(1-alpha[1])//2)]
    
        image_croped = img[c[0]:croped_height+c[0],c[1]:croped_width+c[1]]
        label_croped = label[c[0]:croped_height+c[0],c[1]:croped_width+c[1]]
    
        image_croped = cv2.resize(image_croped, (256,256),interpolation=cv2.INTER_LANCZOS4)
        label_croped = cv2.resize(label_croped, (256,256),interpolation=cv2.INTER_LANCZOS4)
    
        return image_croped, label_croped
    
    
    def flip(self,img,label,alpha):
        image_flipped=cv2.flip(img, 1)
        label_flipped=cv2.flip(label, 1)
        return image_flipped, label_flipped
    
    

class Dataset_loader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, list_length, files_img, files_label, NUM_DEMO, demo_path):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.list_IDs = list_IDs
        self.list_length=list_length
        self.files_img=files_img
        self.files_label=files_label
        self.NUM_DEMO=NUM_DEMO
        
        self.transform_image=transforms.Normalize(0.5,0.5)

        self.demo_path=demo_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        Demo_num,Frame_num=get_traj_index(ID,self.list_length)
        
        image_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'img',self.files_img[Demo_num][Frame_num])
        label_path=os.path.join(self.demo_path,str(self.NUM_DEMO[Demo_num]),'label',self.files_label[Demo_num][Frame_num])
        
        src = cv2.imread(image_path)
         
        img = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        src = cv2.imread(label_path)
        
        label = cv2.resize(src, (256,256),interpolation=cv2.INTER_LANCZOS4)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # label = label/255
        
        augmented_image, augmented_label=self.img_augmentation(img, label)
        augmented_image = augmented_image/255

        return augmented_image, augmented_label
    
    def img_augmentation(self,img, label):
        # return img, label
        return self.contrast(self.brightness(self.noise_level(self.blurriness(self.sharpness(self.crop(self.flip((img,label))))))))
        
    def sharpness(self, img):
        img, label = img
        alpha=np.random.rand()*20+10 #[10,30]
        prob=np.random.rand()
        if prob<0.1:
            blur = cv2.GaussianBlur(img,(0,0),1.0)
            blurr = cv2.GaussianBlur(blur,(0,0),1.0)
            unsharp_image = cv2.addWeighted(blur, alpha+1, blurr, -alpha, 0)
            return unsharp_image.astype('uint8'), label
        else:
            return img, label
    
    def blurriness(self, img):
        img, label = img
        alpha=np.random.rand()*1.25+0.25 #[0.25,1.5]
        prob=np.random.rand()
        if prob<0.1:
            blur_image = cv2.GaussianBlur(img,(0,0),alpha)
            return blur_image.astype('uint8'), label
        else:
            return img, label
    
    def noise_level(self, img):
        img, label = img
        alpha=np.random.rand()*0.04+0.01 #[0.01,0.05]
        prob=np.random.rand()
        if prob<0.1:
            gaussian = np.random.normal(0, alpha, (img.shape[0],img.shape[1]))*255
            noised_image=img+gaussian
            noised_image[noised_image>255]=255
            noised_image[noised_image<0]=0
            return noised_image.astype('uint8'), label
        else:
            return img, label
    
    def brightness(self, img):
        img, label = img
        alpha=np.random.rand()*0.2-0.1
        alpha=int(alpha*255)
        prob=np.random.rand()
        if prob<0.1:
            brightness_image=img+alpha
            brightness_image[brightness_image>255] = 255
            brightness_image[brightness_image<0] = 0
            return brightness_image.astype('uint8'), label
        else:
            return img, label
        
    
    def contrast(self, img):
        img, label = img
        alpha=np.random.rand()*2.5+0.5
        prob=np.random.rand()
        if prob<0.1:
            invGamma = 1 / alpha
            table = [((i / 255) ** invGamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
    
            contrast_image=cv2.LUT(img, table)
    
            return contrast_image.astype('uint8'), label
        else:
            return img, label
        
    def crop(self, img):
        img, label = img
        alpha=np.random.rand(2)*0.2+0.7
        prob=np.random.rand()
        if prob<0.5:
            height, width=img.shape
    
            croped_height=int(height*alpha[0])
            croped_width=int(width*alpha[1])
            
            prob_ = np.random.rand()
            if prob_<0.111:
                c = [0,0]
            elif prob_<0.222:
                c = [0,int(width*(1-alpha[1]))]
            elif prob_<0.333:
                c = [0,int(width*(1-alpha[1])//2)]
            elif prob_<0.444:
                c = [int(height*(1-alpha[0])),0]
            elif prob_<0.555:
                c = [int(height*(1-alpha[0])),int(width*(1-alpha[1]))]
            elif prob_<0.666:
                c = [int(height*(1-alpha[0])),int(width*(1-alpha[1])//2)]
            elif prob_<0.777:
                c = [int(height*(1-alpha[0])//2),0]
            elif prob_<0.888:
                c = [int(height*(1-alpha[0])//2),int(width*(1-alpha[1]))]
            else:
                c = [int(height*(1-alpha[0])//2),int(width*(1-alpha[1])//2)]
                
            # mid = int(height*(1-alpha))
            
            # mid=np.random.randint(int(height*(1-alpha)//2),int(height*(1-alpha)))
            
            image_croped = img[c[0]:croped_height+c[0],c[1]:croped_width+c[1]]
            label_croped = label[c[0]:croped_height+c[0],c[1]:croped_width+c[1]]
        
            image_croped = cv2.resize(image_croped, (256,256),interpolation=cv2.INTER_LANCZOS4)
            label_croped = cv2.resize(label_croped, (256,256),interpolation=cv2.INTER_LANCZOS4)
            
            return image_croped, label_croped
        else:
            return img, label
    
    def flip(self, img):
        img, label = img
        prob=np.random.rand()
        if prob<0.05:
            image_flipped=cv2.flip(img, 1)
            label_flipped=cv2.flip(label, 1)
            return image_flipped, label_flipped
        else:
            return img, label