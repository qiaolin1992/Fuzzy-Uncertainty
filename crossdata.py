from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir,masks_dir,scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = imgs_dir

    def __len__(self):
        return len(self.ids)

    @classmethod#can use the function by class name
    def train_preprocess(cls, pil_img, scale):
        '''
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
'''
        #Rgb_img = pil_img.convert('RGB')
        img_nd = np.array(pil_img)
        #print('img_trans:', img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255


        return img_trans

    @classmethod  # can use the function by class name
    def label_preprocess(cls, pil_img, scale):
        '''
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
'''
        #gray_img=pil_img.convert('L')
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        #print(img_trans[0,:])
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        img_trans=(img_trans>=0.5)*1.0
        #print(img_trans.max())
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        #print('idx',self.masks_dir + idx)
        img_file = idx
        mask_filename=(img_file.split('/')[-1]).replace('.png','_mask.png')

        mask_file = self.masks_dir+mask_filename

        #print('mask_file',mask_file)

       # assert len(mask_file) == 1, \
        #    f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #mask = Image.open(mask_file).resize((256,256))
        #print('mask:',mask_file[0])
        #img = Image.open(img_file).resize((256,256))
        #print('img:',img_file)

        mask = cv2.imread(mask_file,2)
        mask=cv2.resize(mask,(256, 256))
        img = cv2.imread(img_file)
        img=cv2.resize(img,(256,256))

        #assert img.size == mask.size, \
          #  f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.train_preprocess(img, self.scale)
        mask = self.label_preprocess(mask, self.scale)

        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)