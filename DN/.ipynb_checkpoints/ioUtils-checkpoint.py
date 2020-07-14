import numpy as np
import skimage
import skimage.io
import scipy.io as sio
import skimage.transform
import sys

import os
import tensorflow as tf
import cv2
import json
from collections import namedtuple
from tqdm import tqdm

np.random.seed(0)

VGG_MEAN = [103.939, 116.779, 123.68]


def read_mat(path):
    return np.load(path)


def write_mat(path, m):
    np.save(path, m)


def read_ids(path):
    return [line.rstrip('\n') for line in open(path)]

class Batch_Feeder:
    def __init__(self, dataset_path, unet_output_path, indices, subset, batchSize, padWidth=None, padHeight=None, flip=False, keepEmpty=True, train=True, img_shape=(384,384)):
        
        assert subset in ['train', 'val', 'test'], "wrong name of subset"
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._dataset_path = dataset_path
        self._indices = indices
        self._subset = subset
        self._train = train
        self._batchSize = batchSize
        self._padWidth = padWidth
        self._padHeight = padHeight
        self._flip = flip
        self._keepEmpty = keepEmpty
        self.img_shape = img_shape
        self.unet_output_path = os.path.join(unet_output_path, subset)
        mask_path = os.path.join(self._dataset_path, 'polygons.json')
        with open(mask_path) as f:
            self.polygons = json.load(f)
        
        # TO DO: implement shuffling
        # TO DO: support batch wise inference
        

    def set_paths(self):
        self.root = os.path.join(self._dataset_path, self._subset)
        print('scanning {}'.format(self.root))
        self._paths = []
        
        imgs = sorted([i for i in os.listdir(self.root) if i.endswith('.png')])
        gt_DT = sorted([i for i in os.listdir(self.root) if i.endswith('.npy') and 'DT' in i])
        gt_angle = sorted([i for i in os.listdir(self.root) if i.endswith('.npy') and 'angle' in i])
        unet_outputs = sorted([i for i in os.listdir(self.unet_output_path) if i.endswith('.png')])
        
        assert len(imgs)==len(unet_outputs), 'mismatch in imgs and unet++ outputs'
        #actually nested unet++
#         if self._train:

        # TO DO: support batch wise inference

        entry = namedtuple("gt", "index img angle dt unet")
        for index, (img, angle, dt, unet) in enumerate(zip(imgs, gt_angle, gt_DT, unet_outputs)):
            self._paths.append(entry(index, img, angle, dt, unet))

             

            self.shuffle()
#         else:
#             for id in idList:
#                 self._paths.append([id, imageDir + '/' + id + '_leftImg8bit.png',
#                                     ssDir + '/' + id + '_unified_ss.mat'])

        self._numData = len(self._paths)

        if self._numData < self._batchSize:
            self._batchSize = self._numData

    def shuffle(self):
        np.random.shuffle(self._paths)

    def next_batch(self):
        idBatch = []
        
        imageBatch = np.zeros((self._batchSize, self.img_shape[0], self.img_shape[1], 3), dtype=np.int32)
        gtBatch = np.zeros((self._batchSize,  self.img_shape[0], self.img_shape[1], 2), dtype=np.float32)
        ssBatch = np.zeros((self._batchSize,  self.img_shape[0], self.img_shape[1]))
        
        ssUnet = np.zeros((self._batchSize,  self.img_shape[0], self.img_shape[1]))
        
        tmp = 0
        
        if self._train:
            while(len(idBatch) < self._batchSize):
                
                current_tuple = self._paths[self._index_in_epoch]
                
                rgb = self.load_rgb(os.path.join(self.root, current_tuple.img))
        
                #angle = self.load_npy(os.path.join(self.root, current_tuple.angle))
                dt = self.load_npy(os.path.join(self.root, current_tuple.dt))
                
                # not using the calculated gt, check pount 3 of the paper, first equation
                t, t_norm = np.gradient(dt)
                unit_vector_angle = np.stack([t, t_norm], axis=-1)
                
                polygons = self.polygons[current_tuple.img]['polygons']
                mask = self.load_mask(rgb, polygons)
                
                mask_pred =  cv2.imread(os.path.join(self.unet_output_path , current_tuple.unet), cv2.IMREAD_GRAYSCALE)
    
                imageBatch[tmp] = rgb
                gtBatch[tmp] = unit_vector_angle
                ssBatch[tmp] = mask
                ssUnet[tmp] = mask_pred

                idBatch.append(current_tuple.index)
                
                tmp+=1
                if tmp==self._batchSize-1:
                    tmp=0
                self._index_in_epoch += 1
                
                if self._index_in_epoch == self._numData:
                    self._index_in_epoch = 0
                    self.shuffle()

            if self._flip and np.random.uniform() > 0.5:
                for i in range(len(imageBatch)):
                    for j in range(3):
                        imageBatch[i,:,:,j] = np.fliplr(imageBatch[i,:,:,j])

                    weightBatch[i] = np.fliplr(weightBatch[i])
                    ssBatch[i] = np.fliplr(ssBatch[i])
                    ssMaskBatch[i] = np.fliplr(ssMaskBatch[i])

                    for j in range(2):
                        gtBatch[i,:,:,j] = np.fliplr(gtBatch[i,:,:,j])

                    gtBatch[i,:,:,0] = -1 * gtBatch[i,:,:,0]
            return imageBatch, gtBatch, ssBatch, ssUnet
        else:
            pass
            self._index_in_epoch += self._batchSize
            return imageBatch, ssBatch
        
    def total_samples(self):
        return self._numData

    def image_scaling(self, rgb_in):
        if rgb_in.dtype == np.float32:
            rgb_in = rgb_in*255
        elif rgb_in.dtype == np.uint8:
            rgb_in = rgb_in.astype(np.float32)

        # VGG16 was trained using opencv which reads images as BGR, but skimage reads images as RGB
        rgb_out = np.zeros(rgb_in.shape).astype(np.float32)
        rgb_out[:,:,0] = rgb_in[:,:,2] - VGG_MEAN[2]
        rgb_out[:,:,1] = rgb_in[:,:,1] - VGG_MEAN[1]
        rgb_out[:,:,2] = rgb_in[:,:,0] - VGG_MEAN[0]

        return rgb_out

    def pad(self, data):
        if self._padHeight and self._padWidth:
            if data.ndim == 3:
                npad = ((0,self._padHeight-data.shape[0]),(0,self._padWidth-data.shape[1]),(0,0))
            elif data.ndim == 2:
                npad = ((0, self._padHeight - data.shape[0]), (0, self._padWidth - data.shape[1]))
            padData = np.pad(data, npad, mode='constant', constant_values=0)

        else:
            padData = data

        return padData
    
    @staticmethod
    def load_rgb(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def load_mask(img, polygons):
        """
        Transforms polygons of a single image into a 2D binary numpy array
        
        :param img: just to get the corresponding shape of the output array
        :param polygons: - dict
        
        :return mask: numpy array with drawn over and touching polygons
        """
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        for curr_pol in polygons:
            cv2.fillPoly(mask, [np.array(curr_pol, 'int32')], 255)
        return mask
    
    @staticmethod
    def load_npy(path):
        with open(path, 'rb') as f:
            depth = np.load(f)
        return depth
# class Batch_Feeder:
#     def __init__(self, dataset, indices, train, batchSize, padWidth=None, padHeight=None, flip=False, keepEmpty=True):
#         self._epochs_completed = 0
#         self._index_in_epoch = 0
#         self._dataset = dataset
#         self._indices = indices
#         self._train = train
#         self._batchSize = batchSize
#         self._padWidth = padWidth
#         self._padHeight = padHeight
#         self._flip = flip
#         self._keepEmpty = keepEmpty

#     def set_paths(self, idList=None, imageDir=None, gtDir=None, ssDir=None):
#         self._paths = []

#         if self._train:
#             for id in idList:
#                 self._paths.append([id, imageDir + '/' + id + '_leftImg8bit.png',
#                                         gtDir + '/' + id + '_unified_GT.mat',
#                                         ssDir + '/' + id + '_unified_ss.mat'])
#             self.shuffle()
#         else:
#             for id in idList:
#                 self._paths.append([id, imageDir + '/' + id + '_leftImg8bit.png',
#                                     ssDir + '/' + id + '_unified_ss.mat'])

#         self._numData = len(self._paths)

#         if self._numData < self._batchSize:
#             self._batchSize = self._numData

#     def shuffle(self):
#         np.random.shuffle(self._paths)

#     def next_batch(self):
#         idBatch = []
#         imageBatch = []
#         gtBatch = []
#         ssBatch = []
#         ssMaskBatch = []
#         weightBatch = []

#         if self._train:
#             while(len(idBatch) < self._batchSize):
#                 ss = (sio.loadmat(self._paths[self._index_in_epoch][3])['mask']).astype(float)
#                 ssMask = ss
#                 ss = np.sum(ss[:,:,self._indices], 2)

#                 background = np.zeros(ssMask.shape[0:2] + (1,))
#                 ssMask = np.concatenate((ssMask[:,:,[1,2,3,4]], background, ssMask[:,:,[0,5,6,7]]), axis=-1)
#                 ssMask = np.argmax(ssMask, axis=-1)
#                 ssMask = ssMask.astype(float)
#                 ssMask = (ssMask - 4) * 32 # centered at 0, with 0 being background, spaced 32 apart for classes

#                 if ss.sum() > 0 or self._keepEmpty:
#                     idBatch.append(self._paths[self._index_in_epoch][0])

#                     image = (self.image_scaling(skimage.io.imread(self._paths[self._index_in_epoch][1]))).astype(float)
#                     gt = (sio.loadmat(self._paths[self._index_in_epoch][2])['dir_map']).astype(float)
#                     weight = (sio.loadmat(self._paths[self._index_in_epoch][2])['weight_map']).astype(float)

#                     imageBatch.append(self.pad(image))
#                     gtBatch.append(self.pad(gt))
#                     weightBatch.append(self.pad(weight))
#                     ssBatch.append(self.pad(ss))
#                     ssMaskBatch.append(self.pad(ssMask))
#                 else:
#                     pass
#                     # raw_input("skipping " + self._paths[self._index_in_epoch][0])
#                 self._index_in_epoch += 1
#                 if self._index_in_epoch == self._numData:
#                     self._index_in_epoch = 0
#                     self.shuffle()

#             imageBatch = np.array(imageBatch)
#             gtBatch = np.array(gtBatch)
#             ssBatch = np.array(ssBatch)
#             ssMaskBatch = np.array(ssMaskBatch)
#             weightBatch = np.array(weightBatch)

#             if self._flip and np.random.uniform() > 0.5:
#                 for i in range(len(imageBatch)):
#                     for j in range(3):
#                         imageBatch[i,:,:,j] = np.fliplr(imageBatch[i,:,:,j])

#                     weightBatch[i] = np.fliplr(weightBatch[i])
#                     ssBatch[i] = np.fliplr(ssBatch[i])
#                     ssMaskBatch[i] = np.fliplr(ssMaskBatch[i])

#                     for j in range(2):
#                         gtBatch[i,:,:,j] = np.fliplr(gtBatch[i,:,:,j])

#                     gtBatch[i,:,:,0] = -1 * gtBatch[i,:,:,0]
#             return imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, idBatch
#         else:
#             for example in self._paths[self._index_in_epoch:min(self._index_in_epoch+self._batchSize, self._numData)]:
#                 imageBatch.append(self.pad((self.image_scaling(skimage.io.imread(example[1]))).astype(float)))
#                 idBatch.append(example[0])
#                 ss = (sio.loadmat(example[2])['mask']).astype(float)
#                 ssMask = ss
#                 ss = np.sum(ss[:, :, self._indices], 2)
#                 background = np.zeros(ssMask.shape[0:2] + (1,))
#                 ssMask = np.concatenate((ssMask[:,:,[1,2,3,4]], background, ssMask[:,:,[0,5,6,7]]), axis=-1)
#                 ssMask = np.argmax(ssMask, axis=-1)
#                 ssMask = ssMask.astype(float)
#                 ssMask = (ssMask - 4) * 32 # centered at 0, with 0 being background, spaced 32 apart for classes
#                 ssBatch.append(self.pad(ss))
#                 ssMaskBatch.append(self.pad(ssMask))
#             imageBatch = np.array(imageBatch)
#             ssBatch = np.array(ssBatch)
#             ssMaskBatch = np.array(ssMaskBatch)

#             self._index_in_epoch += self._batchSize
#             return imageBatch, ssBatch, ssMaskBatch, idBatch

    # def total_samples(self):
    #     return self._numData

    # def image_scaling(self, rgb_in):
    #     if rgb_in.dtype == np.float32:
    #         rgb_in = rgb_in*255
    #     elif rgb_in.dtype == np.uint8:
    #         rgb_in = rgb_in.astype(np.float32)

    #     # VGG16 was trained using opencv which reads images as BGR, but skimage reads images as RGB
    #     rgb_out = np.zeros(rgb_in.shape).astype(np.float32)
    #     rgb_out[:,:,0] = rgb_in[:,:,2] - VGG_MEAN[2]
    #     rgb_out[:,:,1] = rgb_in[:,:,1] - VGG_MEAN[1]
    #     rgb_out[:,:,2] = rgb_in[:,:,0] - VGG_MEAN[0]

    #     return rgb_out

    # def pad(self, data):
    #     if self._padHeight and self._padWidth:
    #         if data.ndim == 3:
    #             npad = ((0,self._padHeight-data.shape[0]),(0,self._padWidth-data.shape[1]),(0,0))
    #         elif data.ndim == 2:
    #             npad = ((0, self._padHeight - data.shape[0]), (0, self._padWidth - data.shape[1]))
    #         padData = np.pad(data, npad, mode='constant', constant_values=0)

    #     else:
    #         padData = data

    #     return padData

