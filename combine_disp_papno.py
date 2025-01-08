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
import matplotlib.pyplot as plt
# from data.imgutils import rescale_size, imresize, imrescale, imflip, impad, impad_to_multiple
import sys
import cv2
import pyshine as ps



cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    
    resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale
# COCO_LABEL = [1,  2,  3,  4,  5,  6,  7,  8,
#                    9, 10, 11, 13, 14, 15, 16, 17,
#                   18, 19, 20, 21, 22, 23, 24, 25,
#                   27, 28, 31, 32, 33, 34, 35, 36,
#                   37, 38, 39, 40, 41, 42, 43, 44,
#                   46, 47, 48, 49, 50, 51, 52, 53,
#                   54, 55, 56, 57, 58, 59, 60, 61,
#                   62, 63, 64, 65, 67, 70, 72, 73,
#                   74, 75, 76, 77, 78, 79, 80, 81,
#                   82, 84, 85, 86, 87, 88, 89, 90]

# COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
#                    9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
#                   18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
#                   27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
#                   37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
#                   46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
#                   54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
#                   62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
#                   74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
#                   82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

# COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#                 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
COCO_LABEL=[]
COCO_LABEL_MAP=dict()
COCO_CLASSES=[]


annota = json.load(open('/home/cuonghoang/Downloads/stuff_annotations_trainval2017/annotations/stuff_val2017.json'))

for c,i in enumerate(annota['categories']):
    COCO_LABEL.append(i['id'])
    COCO_LABEL_MAP[i['id']]=(c+1)
    COCO_CLASSES.append(i['name'])

from PIL import Image
def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((50000, 3), dtype=int)
    ind = np.arange(50000, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    # print('colormap',colormap.shape)
    # sys.exit()
    return colormap
color=create_pascal_label_colormap()
def show_result_ins(img,
                    result,
                    score_thr=0.3,
                    sort_by_density=False):
    
    name1=(img.split('/'))[-1]
    # print('img',img)
    if isinstance(img, str):
        img = cv.imread(img)

    img = cv.resize(img, (500, 375))
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    # seg_label = seg_label.cpu().numpy().astype(np.uint8)
    seg_label = seg_label#.astype(np.uint8)

    # print('vis_inds123',seg_label.shape)
    cate_label = cur_result[1]
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

    # print('vis_inds',seg_label.shape)
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
    # np.random.seed(12)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    #img_show = None
    
    for i in [1,2]:
        dem=0
        image_show=0
        for idx in range(num_mask):
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
            realclass = cur_cate#COCO_LABEL[cur_cate]
            # print('realclass',realclass)
            # realclass=2
            try:
                name_idx = COCO_LABEL_MAP[realclass]
                # label_text = COCO_CLASSES[name_idx-1]
                label_text = COCO_CLASSES[name_idx-1]
            except:
                if realclass==1:
                    label_text='grass1'
                if realclass==2:
                    label_text='elephant'
                if realclass==3:
                    label_text='elephant'

            label_text1=label_text
            dem=dem+1
            label_text=str(dem)+label_text


            if label_text1=='sky-other':
                label_text1='sky'

            
            print(label_text)
        
            
            pos_x=0
            pos_y=0

        

            # if label_text1=='air_conditioner' :
            #     pos_y=10
            if label_text1=='grass' or label_text1=='grass1':
                color_mask=np.array([144, 238, 144], dtype=np.uint8)
            if label_text1=='mountain' or label_text1=='mountain1':
                color_mask=np.array([0, 100, 0], dtype=np.uint8)
            if label_text1=='clouds':
                color_mask=np.array([220, 220, 220], dtype=np.uint8)
            if label_text1=='sky':
                color_mask=np.array([173, 216, 230], dtype=np.uint8)
            if label_text1=='sea':
                color_mask=np.array([64, 224, 208], dtype=np.uint8)

        
            # try:
            #     label_text1=dict1[label_text1]
            # except:
            #     pass
            # if i==1:
            print('label_text1',label_text1)
            if label_text1!='other':
                


                center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)

                
                vis_pos = (max(int(center_x) - 10, 0)-pos_x, int(center_y)-pos_y)
                if i==1:
                    if label_text1 in ['elephant','grass1','clouds']:
                        # print('color_mask',color_mask,label_text1)
                        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5
                        # cv.putText(img_show, label_text1, vis_pos,
                                        # cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
                if i==2:
                    if label_text1 in ['elephant','grass1','clouds']:
                        if label_text1=='grass1':
                            label_text1='grass'
                        ps.putBText(img_show,label_text1,text_offset_x=vis_pos[0],text_offset_y=vis_pos[1],vspace=3,hspace=3, font_scale=0.4,background_RGB=(255,255,255),text_RGB=(60,60,60),thickness=0,font=cv.FONT_HERSHEY_SIMPLEX)
            
            # print('label_text',label_text)
            # if label_text in allow:
                # if label_text1=='aerosol_can':
                #     label_text1='mirror'
                # if label_text1=='bathrobe':
                #     label_text1='towel'

                # if label_text1=='yogurt':
                #     label_text1='bowl'
                # if i==1:
                #     img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5


                    # image_show=image_show+cur_mask*name_idx
                
                # center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
                # vis_pos = [max(int(center_x) - 10, 0), int(center_y)]

                # if i==2:
                    # ps.putBText(img_show,label_text1,text_offset_x=vis_pos[0],text_offset_y=vis_pos[1],vspace=3,hspace=3, font_scale=0.4,background_RGB=(255,255,255),text_RGB=(60,60,60),thickness=0,font=cv.FONT_HERSHEY_SIMPLEX)
        
    
    
    # plt.imshow(image_show, cmap='hot', interpolation='nearest')
    # plt.show()
    display=color[image_show].astype(np.uint8)
 
    # sys.exit()
    # asd='sme_'+name1
    # Image.fromarray(display).save(asd)
 
    return img_show

import pycocotools.mask as mask_util
from pycocotools.coco import COCO

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
a69 = json.load(open('/home/cuonghoang/Downloads/stuff_annotations_trainval2017/annotations/stuff_val2017.json'))['images']
# ann=a['annotations']
# imgsinfo=json.load(open("data/coco/annotations/instances_val2017.json",'r'))
# coco = COCO("data/coco/annotations/instances_val2017.json")
# coco = COCO("/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/annotations/instances_val2017.json")

coco = COCO("/home/cuonghoang/Downloads/stuff_annotations_trainval2017/annotations/stuff_val2017.json")
list_id=set()

coco1 = COCO("/home/cuonghoang/Downloads/labels_my-project-name_2024-08-19-12-14-42.json") 
# a64 = json.load(open('lvis_masks.json'))

# for k in a64:
#      list_id.add(k['image_id'])


# from tqdm import tqdm

imgpath="/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/val2017/"
# imgpath="/home/cuonghoang/Desktop/codedict/pytorch_solov2-master/data/coco/train2017/"

name='000000024919.jpg'   #000000006894  000000024919 000000027972  000000257478
for l in a69:
     if os.path.basename(l['coco_url'])==name:  
        img_id=l['id']


ann_ids = coco.getAnnIds(imgIds=[img_id])
        
ann_info_anno = coco.loadAnns(ann_ids)


ann_ids1 = coco1.getAnnIds(imgIds=[1])
        
ann_info_tool = coco1.loadAnns(ann_ids1)
# ann_ids = coco.getAnnIds(imgIds=[img_id])
# print('ann_info_anno',ann_info_anno)
# exit()
# sys.exit()
# ann_info = coco.loadAnns(ann_ids)
# print('ann_info',ann_info)
# sys.exit()
# info = coco.loadImgs(img_id)[0]

# show=a64 #a69
# ann_info=list(filter(lambda person: person['image_id'] == img_id, show))#[0]
# print('ann_info',ann_info)
# sys.exit()
a=[]
b=[]
c=[]
num_pix=[]

# ###result###
# for k in ann_info:
#     # print('kkk',k)
#     # mask=segToMask(k['segmentation'],info['height'],info['width'])
#     mask=mask_util.decode([k['segmentation']])#[:,:,0].astype(np.uint8)
#     # print('mask',mask)
#     # sys.exit()
#     mask=np.expand_dims(mask, axis=0)
#     # print('mask',mask.shape)
#     # if COCO_CLASSES[COCO_LABEL.index(k['category_id'])] not in ['tennis racket']:
#     a.append(mask)
#     b.append(int(k['category_id']))
#     c.append(k['score'])
#     num_pix.append(np.sum(mask))
    # print(1231231231,mask.shape,np.unique(mask))
# print('ann_info_anno',ann_info_anno)

# # # ##anno###
for k in ann_info_anno:
   
    # mask=segToMask(k['segmentation'],info['height'],info['width'])
    # print('kk',k)
    # exit()
    mask1 = coco.annToMask(k)
    # print('mask1',mask1)
    # exit()
    mask1=np.expand_dims(mask1, axis=0)
    mask1=np.expand_dims(mask1, axis=-1)
    # print('mask1',mask1.shape)

    cur_cate = k['category_id']
    realclass = cur_cate#COCO_LABEL[cur_cate]
    name_idx = COCO_LABEL_MAP[realclass]
    label_text = COCO_CLASSES[name_idx-1]
    # if label_text=='lightbulb':
        # print('k12')
    a.append(mask1)
    b.append(k['category_id'])
    c.append(0.7)
    num_pix.append(np.sum(mask1))



##tool###
for k in ann_info_tool:
    # print('k')
    # mask=segToMask(k['segmentation'],info['height'],info['width'])
    mask1 = coco1.annToMask(k)
   
    mask1=np.expand_dims(mask1, axis=0)
    mask1=np.expand_dims(mask1, axis=-1)
    # print('mask1',mask1.shape)

    
    a.append(mask1)
    b.append(k['category_id'])
    c.append(0.7)
    num_pix.append(np.sum(mask1))

# print('num_pix',num_pix)

new_index=list(np.argsort(np.array(num_pix)))
print('new_index',new_index)
# new_index=[1,5,4,0,3,2]
a = [a[i] for i in new_index]
b = [b[i] for i in new_index]
c = [c[i] for i in new_index]

# num_pix,a, b,c = zip(*sorted(zip(num_pix,a, b,c),reverse=True)).all()

# print('num_pix',num_pix)

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
imgpath=imgpath+'/'+name #img_id+'.jpg'
# sys.exit()
# name=img_id+'.jpg'
# print('imgpath',imgpath)
print('name',name)
img_show = show_result_ins(imgpath,seg_result)
cv.imwrite('out'+name, img_show)