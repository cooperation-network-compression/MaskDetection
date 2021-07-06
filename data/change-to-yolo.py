import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

sets=[('train'), ('val')]

classes = ["face_mask", "face"]

# XML坐标格式转换成yolo坐标格式 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# 标记文件格式转换
def convert_annotation(chose,image_id):
    # E:\Detection\MaskDetect-YOLOv4-PyTorch-master\datasets\train
    in_file = open('/home/featurize/data/datasets/%s/Annotations/%s.xml'%(chose,image_id))
    out_file = open('/home/featurize/data/datasets/%s/labels/%s.txt'%(chose,image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        if w == 0 or h == 0:
            continue
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        print(image_id)
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#wd = getcwd()

for image_set in sets:
    if not os.path.exists('/home/featurize/data/datasets/%s/labels'%(image_set)):
        os.makedirs('/home/featurize/data/datasets/%s/labels'%(image_set))
    image_ids = open('/home/featurize/work/yolov5-5.0/data/%s.txt'%(image_set)).read().strip().split()
    #list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        #list_file.write('datasets\\%s\\JPEGImages\\%s.jpg\n'%(image_set, image_id))
        convert_annotation(image_set, image_id)
    #list_file.close()



