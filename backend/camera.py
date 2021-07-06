import os
import cv2
from backend.base_camera import BaseCamera
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from utils.datasets import *
import threading

from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import plot_one_box
from flask import Flask,url_for,jsonify
from backend.flask_id2name import id2name

import json
import numpy as np
from backend.predict import predict
from pathlib import Path
import easydict
from backend.warning import playsound
 


with open('./backend/flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)

# 选择设备
device = select_device(opt['device'])
# 加载模型
model = attempt_load(opt['weights'], map_location=device)

class Camera(BaseCamera):

    # video_source = 0
    # palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    #video_source = 0
    def __init__(self):
        # if os.environ.get('OPENCV_CAMERA_SOURCE'):
        #     print('走了吗')
        #     Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        # camera = cv2.VideoCapture(Camera.video_source)
        # if not camera.isOpened():
        #     raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            # _, img = camera.read()
            # if (img.mode != 'RGB'):
            #     img = img.convert("RGB")
            # save_path = str(Path(opt['source']) / Path("img4predict.jpg")) # 保存路径
            # img.save(save_path) # 保存文件
            # img.save("./frontend/static/images/img4predict.jpg")  

            out, source, view_img, save_img, save_txt, imgsz = \
                opt['output'], '0', opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']
            
            webcam = source.isnumeric()
            # if os.path.exists(out):
                #     shutil.rmtree(out)  # delete output folder
                # os.makedirs(out)  # make new output folder
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # im0_shape = img.shape # 记下原始图片的尺寸
            #print('im0_shape = %s \n' % str(im0_shape))

            # Load model
            # model = attempt_load(weights, map_location=device)  # load FP32 model
            imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
            if half:
                model.half()  # to FP16

            # Set Dataloader
            dataset = LoadStreams(source, img_size=imgsz)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            t0 = time.time()
            frametimes = 0
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filetime = str(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))).replace(" ","")
            filesort = '.mp4'
            video = 'static/static/real/'+filetime + filesort
            print(video)
            # imgInfo = im0.shape
            # size = (imgInfo[1],imgInfo[0])
            # print(size)
            videoWrite = cv2.VideoWriter(video, fourcc, 30.0, (640,480))
            if not videoWrite.isOpened():
                print('videoWrite创建失败')
            for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                # print('path:{0}'.format(path))
                # print('im0s:{0}'.format(im0s))
                # print('im0s类型:{0}'.format(type(im0s)))
                print(frame_idx)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                t1 = time_synchronized()
                
                # 前向推理
                pred = model(img, augment=opt['augment'])[0] 
                # Apply NMS（非极大抑制）
                pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
            
                t2 = time_synchronized()

                speed = t2 - t1 

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    #p, s, im0 = path, '', im0s
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    #save_path = str(Path(out) / Path(p).name) # 保存路径
                    #txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string
                    #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        #for c in det[:, -1].unique():  #probably error with torch 1.5
                        for c in det[:, -1].detach().unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %s, ' % (n, names[int(c)])  # add to string

                        # Write results
                        boxes_detected = [] #检测结果
                        for *xyxy, conf, cls in reversed(det):
                            xyxy_list = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  
                            boxes_detected.append({"name": id2name[int(cls.item())],
                                            "conf": str(conf.item()),
                                            "bbox": [int(xyxy_list[0]), int(xyxy_list[1]), int(xyxy_list[2]), int(xyxy_list[3])],
                                            "speed":float(speed),
                                            "detail":str(s)
                                            })
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                videoWrite.write(im0)

                # print(s)
                # print('有没有no-mask'+str(s.find('no-mask') != -1))
                # print('有没有过去10帧'+str((frametimes%10) == 0))
                # print('两个的判断结果'+str(s.find('no-mask') != -1 and (frametimes%10) == 0))
                if s.find('no-mask') != -1 and (frametimes%10) == 0:
                    threading.Thread(target=playsound).start()
                frametimes += 1
                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', im0)[1].tobytes()
                    

            # convert to numpy array.
            # img_arr = np.array(img)
            
            results = {"results": boxes_detected}
            print(results)
        # for _, item in results.items():
        #         for i in range(len(item)):
        #             for key, value in item[i].items():
        #                 if key == 'name':
        #                     name = value
        #                 if key == 'conf':
        #                     conf = float(value)
        #                 if key =='bbox':
        #                     bbox = value
        #                     x1, y1, x2, y2 = [int(i) for i in bbox]
                    
        #             label = '%s %.2f' % (name, conf)
        #             label += '%'
        #             color = [random.randint(0, 255) for _ in range(3)]
        #             t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        #             cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
        #             cv2.putText(img, label, (x1, y1 +
        #                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

            
            
            
            

    
    
    
    