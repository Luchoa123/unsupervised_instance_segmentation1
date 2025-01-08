# from data.config import cfg, process_funcs_dict
# from data.coco import CocoDataset
# from data.loader import build_dataloader
# from modules.solov2 import SOLOV2
import torch.optim as optim
import time
import argparse
import torch
from torch.nn.utils import clip_grad
import pycocotools.mask as mask_util
import numpy as np
import cv2 as cv
# from data.compose import Compose
from glob import glob
import pycocotools.mask as maskutil
import json
import os
from skimage.draw import polygon
from scipy import ndimage
from data.imgutils import rescale_size, imresize, imrescale, imflip, impad, impad_to_multiple
import sys
COCO_LABEL = [1,  2,  3,  4,  5,  6,  7,  8,
                   9, 10, 11, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25,
                  27, 28, 31, 32, 33, 34, 35, 36,
                  37, 38, 39, 40, 41, 42, 43, 44,
                  46, 47, 48, 49, 50, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 60, 61,
                  62, 63, 64, 65, 67, 70, 72, 73,
                  74, 75, 76, 77, 78, 79, 80, 81,
                  82, 84, 85, 86, 87, 88, 89, 90]

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# COCO_LABEL=[]
# COCO_CLASSES=[]
# COCO_LABEL_MAP=dict()
# for c,k in enumerate(a123['categories']):
#     COCO_CLASSES.append(k['name'])
#     COCO_LABEL_MAP[int(k['id'])]=c+1
#     COCO_LABEL.append(int(k['id']))

# COCO_CLASSES=tuple(COCO_CLASSES)
# print('COCO_CLASSES',COCO_CLASSES[1])
# print('COCO_LABEL',COCO_LABEL)
# sys.exit()
def show_result_ins(img,
                    result,
                    method,
                    score_thr=0.3,
                    sort_by_density=False):
    if isinstance(img, str):
        img = cv.imread(img)

    img = cv.resize(img, (500, 375))
    img_show = img.copy()
    h, w, _ = img.shape
    # print('hh',h,w)
    cur_result = result[0]
    seg_label = cur_result[0]
    # seg_label = seg_label.cpu().numpy().astype(np.uint8)
    seg_label = seg_label#.astype(np.uint8)
    cate_label = cur_result[1]
    # bbox=cur_result[2]
    # print(cate_label)
    # sys.exit()
    # cate_label = cate_label.cpu().numpy()
    # cate_label = cate_label.numpy()
    # print(cur_result[2])
    # sys.exit()
    score = np.array(cur_result[2])#.cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    # print('hahaha',seg_label.shape)
    # sys.exit()
    num_mask = seg_label.shape[0]

    # print('vis_inds',vis_inds)
    # print(cate_label)
    # loc=np.where(vis_inds==True)[0]
    # print(loc)
    # sys.exit()
    # print(len(cate_label),cate_label)
    cate_label=np.array(cate_label)
    cate_label = cate_label[vis_inds]
    # cate_score = score[vis_inds]
    # print('seg_label',seg_label.shape)
   
    # sys.exit()
    # np.random.seed(52)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    #img_show = None
    for i in [1,2]:
        for idx in range(num_mask):
            idx1=idx
            idx = -(idx+1)
            cur_mask = seg_label[idx, :, :]*1
            # print('cur_mask',cur_mask.shape,cur_mask)
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)

            cur_cate = cate_label[idx]
            # print('cur_cate',cur_cate)
            # print('COCO_LABEL',len(COCO_LABEL))
            # print(COCO_LABEL)
            # print('cur_cate',cur_cate)
            # sys.exit()
            realclass = cur_cate#COCO_LABEL[cur_cate]
        
            # cur_score = cate_score[idx]

            name_idx = COCO_LABEL_MAP[realclass]
            label_text = COCO_CLASSES[name_idx-1]
            # if label_text!='cat':
            # if method=='cutler' and idx!=1:
            if idx1 in [0]:
                if i==1:
                    img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

                #当前实例的类别
                # print('cate_label',cate_label)
                # print(idx)

                # if idx==-2:
                #      label_text='cup'
                # print('label_text',idx, label_text)
                # label_text += '|{:.02f}'.format(cur_score)
                center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
                vis_pos = (max(int(center_x) - 10, 0), int(center_y))
                # cv.putText(img_show, label_text, vis_pos,
                                # cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))  # green
                c1,c2,c3=color_mask[0]
                coords = np.column_stack(np.where(cur_mask_bool > 0))
            
                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0)
                if i==2:
                    cv.rectangle(img_show, (int(x1), int(y1)), (int(x2),int( y2)), (int(c1),int(c2),int(c3)), 2)
 
    return img_show

import pycocotools.mask as mask_util

def segToMask( S, h, w ):
         """
         Convert polygon segmentation to binary mask.
         :param   S (float array)   : polygon segmentation mask
         :param   h (int)           : target mask height
         :param   w (int)           : target mask width
         :return: M (bool 2D array) : binary mask
         """
         M = np.zeros((h,w), dtype=np.bool)
         for s in S:
             N = len(s)
             rr, cc = polygon(np.array(s[1:N:2]).clip(max=h-1), \
                              np.array(s[0:N:2]).clip(max=w-1)) # (y, x)
             M[rr, cc] = 1
         return M
from pycocotools.coco import COCO

# /home/cuonghoang/Desktop/codedict/pytorch_solov2-master
# a = json.load(open('/home/cuonghoang/Desktop/codedict/solov2-voc/data/coco/annotations/instances_val2017.json'))
# a = json.load(open('/home/cuonghoang/Desktop/external_PASCAL_VOC/PASCAL_VOC/pascal_val2007.json'))
a69 = json.load(open('/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/annotations/instances_val2017.json'))['images']
# ann=a['annotations']
# imgsinfo=json.load(open("data/coco/annotations/instances_val2017.json",'r'))
# coco = COCO("coco_cutler.json")
coco = COCO("/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/annotations/instances_val2017.json")

list_id=set()

# method='solo'  
# method='freesolo'  
# method='solo'  
# method=['solo','freesolo','cutler']
method=['solo']
for k1 in method:
    print('method',k1)
    a64 = json.load(open('coco_'+k1+'.json'))


    # for k in a64:
    #      list_id.add(k['image_id'])


    # from tqdm import tqdm

    imgpath="/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/val2017/"

    name='000000206994.jpg'  #000000060770  000000010363  000000079031  000000011149
    for l in a69:
        if l['file_name']==name:  
            img_id=l['id']

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    # print('ann_ids',ann_ids)
    # sys.exit()
    # ann_info = coco.loadAnns(ann_ids)
    # print('ann_info',ann_info)
    # sys.exit()
    # info = coco.loadImgs(img_id)[0]

    show=a64 #a69
    ann_info=list(filter(lambda person: person['image_id'] == img_id, show))#[0]
    # print('ann_info',ann_info)
    # sys.exit()
    a=[]
    b=[]
    c=[]
    num_pix=[]
    for k in ann_info:
        # print('kkk',k)
        # mask=segToMask(k['segmentation'],info['height'],info['width'])
        mask=mask_util.decode([k['segmentation']])#[:,:,0].astype(np.uint8)
        # print('mask',mask)
        # sys.exit()
        mask=np.expand_dims(mask, axis=0)
        # if COCO_CLASSES[COCO_LABEL.index(k['category_id'])] not in ['tennis racket']:
        a.append(mask)
        b.append(int(k['category_id']))
        c.append(k['score'])
        num_pix.append(np.sum(mask))
        # print(1231231231,mask.shape,np.unique(mask))




    new_index=list(np.argsort(np.array(num_pix)))
    a = [a[i] for i in new_index]
    b = [b[i] for i in new_index]
    c = [c[i] for i in new_index]


    a=np.concatenate((a), axis=0)
    seg_result1=[]
    seg_result=[]



    seg_result1.append(a)
    seg_result1.append(b)
    seg_result1.append(c)

    seg_result.append(seg_result1)
    # print('imgpath',imgpath)
    # print(img_id)
    # print(info['file_name'])
    # sys.exit()
    imgpath=imgpath+name #img_id+'.jpg'
    # sys.exit()
    # name=img_id+'.jpg'
    # print('imgpath',imgpath)
    # print('name',name)
    img_show = show_result_ins(imgpath,seg_result,k1)
    import cv2
    # display = cv2.resize(img_show, (480,480), interpolation = cv2.INTER_AREA)
    cv.imwrite(k1+'_'+name, img_show)







