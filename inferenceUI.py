import sys
sys.path.append('./CRAFTpytorch/')
# sys.path.append('./CRAFTpytorch/test.py')
import os
import time
import argparse
import gradio as gr
import json

import torch
import math
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import ImageFont, ImageDraw, Image

import cv2
from skimage import io
import numpy as np
import craft_utils
# import test
import imgproc
import file_utils
import json
import zipfile
import pandas as pd
import CRAFTpytorch.test as test
# import test

from CRAFTpytorch.test import copyStateDict
# from test import copyStateDict
from craft import CRAFT

from collections import OrderedDict

from pathlib import Path
from numpy import random
sys.path.append('./yolov5/')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
# DETECTION _____________________________________________________
def load_detection_model():
  parser = argparse.ArgumentParser(description='CRAFT Text Detection')
  parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
  parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
  parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
  parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
  parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
  parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
  parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
  parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
  parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
  parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
  parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
  parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
  # args = parser.parse_args(["--trained_model=/content/CRAFTpytorch/basenet/craft_mlt_25k.pth","--test_folder="+impath,"--refine", "--refiner_model=/content/CRAFTpytorch/basenet/craft_refiner_CTW1500.pth"])
  args = parser.parse_args(["--trained_model=./CRAFTpytorch/basenet/craft_mlt_25k.pth","--refine", "--refiner_model=./CRAFTpytorch/basenet/craft_refiner_CTW1500.pth"])
  net = CRAFT()     # initialize
  print('Loading weights from checkpoint (' + args.trained_model + ')')
  if args.cuda:
      net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
  else:
      net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

  if args.cuda:
      net = net.cuda()
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = False

  net.eval()

  # LinkRefiner
  refine_net = None
  if args.refine:
      from refinenet import RefineNet
      refine_net = RefineNet()
      print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
      if args.cuda:
          refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
          refine_net = refine_net.cuda()
          refine_net = torch.nn.DataParallel(refine_net)
      else:
          refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

      refine_net.eval()
      # args.poly = True
  return net,refine_net,args


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
def infer_detection(image,image_name,net,refine_net,args):
  #CRAFT
  data={}
  t = time.time()

  # load data
  # image = imgproc.loadImage(image_path)
  # image_name=int(os.path.relpath(image_path, start).replace('.jpg',''))
  bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)
  bbox_score={}
  index=0
  for box,conf in zip(bboxes,det_scores):
    bbox_score[str(index)]={}
    bbox_score[str(index)]['detconf']=str(conf)
    bbox_score[str(index)]['box']=[]
    for coors in box:
      temp=[str(coors[0]),str(coors[1])]
      bbox_score[str(index)]['box'].append(temp)
    index+=1
  data[image_name]=bbox_score
  if not os.path.isdir('./Results'):
    os.mkdir('./Results')
  # data.to_csv('./Results_csv/data.csv', sep = ',', na_rep='Unknown')
  # print(data)
  with open('./Results/data.json', 'w') as jsonfile:
    json.dump(data, jsonfile,sort_keys=True)
    jsonfile.close()
  print("elapsed time : {}s".format(time.time() - t))


# RECOGNITION__________________________
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
def load_recognition_model():
  #chuan bi ocr predict model
  config = Cfg.load_config_from_name('vgg_seq2seq')
  config['cnn']['pretrained']=False
  config['device'] = 'cuda:0'
  config['predictor']['beamsearch']=False
  recognizer = Predictor(config)
  return recognizer

import shutil
def crop(pts, image):

  """
  Takes inputs as 8 points
  and Returns cropped, masked image with a white background
  """
  for i in pts:
    if (i[0]<0):
      i[0]=0
    if(i[1]<0):
      i[1]=0
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  #print('x,y,w,h:',x,y,w,h)
  #print(image)
  cropped = image[y:y+h, x:x+w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(cropped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(cropped, cropped, mask=mask)
  bg = np.ones_like(cropped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst

  return dst2
def generate_words(image_name, score_bbox, image,recognizer,tsvdata):

  #score_bbox: {'0': {'detconf': '0.886273', 'box': [['604.8', '116.8'], ['737.6', '116.8'], ['737.6', '209.6'],...
  num_bboxes = len(score_bbox)

  for num in range(num_bboxes): #duyet qua moi bbox trong 1 image
    bbox_coords = score_bbox[str(num)]['box']
    if(bbox_coords):
      l_t = float(bbox_coords[0][0])
      t_l = float(bbox_coords[0][1])
      r_t = float(bbox_coords[1][0])
      t_r = float(bbox_coords[1][1])
      r_b = float(bbox_coords[2][0])
      b_r = float(bbox_coords[2][1])
      l_b = float(bbox_coords[3][0])
      b_l = float(bbox_coords[3][1])
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      #print('pts:',pts)
      if np.all(pts) > 0:
        # break
        word = crop(pts, image)
        img=Image.fromarray(word)
        trans=recognizer.predict(img)
        # print(str(num),':',trans, type(trans))
        coords=tsvdata[image_name][str(num)]['box']
        coords=["1",coords[0][0],coords[0][1],coords[1][0],coords[1][1],coords[2][0],coords[2][1],coords[3][0],coords[3][1]]
        tsvdata[image_name][str(num)]['trans']=trans
        # folder = '/'.join( image_name.split('/')[:-1])
        # folder=image_name


        # #CHANGE DIR
        # if not os.path.isdir('./cropped_words'):
        #   os.mkdir('./cropped_words')
        # dir = './cropped_words/'

        # if not os.path.isdir(os.path.join(dir + folder)):
        #   os.mkdir(os.path.join(dir + folder))
        # dir=dir+folder+'/'

        # try:
        #   # print(image_name)
        #   file_name = os.path.join(dir + image_name+'_'+str(num))
        #   # cv2.imwrite(file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l), word)
        #   cv2.imwrite(file_name+'.jpg',word)
        #   #print('Image saved to '+file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l))
        # except:
        #   continue


def crop_OCR(recognizer, image,image_name):
  data=json.load(open('./Results/data.json')) #PATH TO CSV
  # print(data)

  # start = './frames' #PATH TO TEST IMAGES

  # for image_name in data:
    #print(str(os.path.join(start, data['image_name'][image_num])))
    # image = cv2.imread(os.path.join(start,image_name+'.jpg'))
  score_bbox = data[image_name]
  generate_words(image_name, score_bbox, image,recognizer,data)
  with open('./Results/data.json', 'w') as jsonfile:
    json.dump(data, jsonfile)
    jsonfile.close()
  # shutil.rmtree(start)


# yolo_________________________
def load_yolo():
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
  parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
  parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
  parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--view-img', action='store_true', help='display results')
  parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
  parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
  parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
  parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
  parser.add_argument('--augment', action='store_true', help='augmented inference')
  parser.add_argument('--update', action='store_true', help='update all models')
  parser.add_argument('--project', default='runs/detect', help='save results to project/name')
  parser.add_argument('--name', default='exp', help='save results to project/name')
  parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
  # opt = parser.parse_args()
  opt = parser.parse_args(["--weights=./yolov5/runs/train/best.pt","--img=320", "--conf=0.5","--save-txt","--exist-ok"])
  check_requirements(exclude=('pycocotools', 'thop'))
  print(opt)
  check_requirements(exclude=('pycocotools', 'thop'))
  weights, view_img, save_txt, imgsz = opt.weights, opt.view_img, opt.save_txt, opt.img_size

  # Initialize
  set_logging()
  device = select_device(opt.device)
  half = device.type != 'cpu'  # half precision only supported on CUDA

  # Load model
  model = attempt_load(weights, map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  if half:
      model.half()  # to FP16
  if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
  return opt,model,stride


def detect(opt,model,stride,save_img=False):
  source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
  save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
  webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
      ('rtsp://', 'rtmp://', 'http://', 'https://'))

  # Directories
  save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
  (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

  # Initialize
  device = select_device(opt.device)
  half = device.type != 'cpu'  # half precision only supported on CUDA
  # Set Dataloader
  vid_path, vid_writer = None, None
  dataset = LoadImages(source, img_size=imgsz, stride=stride)

  # Get names and colors
  names = model.module.names if hasattr(model, 'module') else model.names
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

  # Run inference
  t0 = time.time()
  for path, img, im0s, vid_cap in dataset:
      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      # Inference
      t1 = time_synchronized()
      pred = model(img, augment=opt.augment)[0]

      # Apply NMS
      pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
      t2 = time_synchronized()

      # Apply Classifier
      # if classify:
      #     pred = apply_classifier(pred, modelc, img, im0s)

      # Process detections
      for i, det in enumerate(pred):  # detections per image
          p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

          p = Path(p)  # to Path
          save_path = str(save_dir / p.name)  # img.jpg
          txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
          s += '%gx%g ' % img.shape[2:]  # print string
          gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
          if len(det):
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

              # Print results
              for c in det[:, -1].unique():
                  n = (det[:, -1] == c).sum()  # detections per class
                  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

              # Write results
              for *xyxy, conf, cls in reversed(det):
                  if save_txt:  # Write to file
                      xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                      line = (cls, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf) if opt.save_conf else (cls, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))  # label format
                      # print(line)
                      with open(txt_path + '.txt', 'a') as f:
                          f.write(('%g ' * len(line)).rstrip() % line + '\n')

                  if save_img or view_img:  # Add bbox to image
                      label = f'{names[int(cls)]} {conf:.2f}'
                      plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

          # Print time (inference + NMS)
          # print(f'{s}Done. ({t2 - t1:.3f}s)')

          # Stream results
          if view_img:
              cv2.imshow(str(p), im0)
              cv2.waitKey(1)  # 1 millisecond

          # Save results (image with detections)
          if save_img:
              if dataset.mode == 'image':
                  cv2.imwrite(save_path, im0)
              else:  # 'video' or 'stream'
                  if vid_path != save_path:  # new video
                      vid_path = save_path
                      if isinstance(vid_writer, cv2.VideoWriter):
                          vid_writer.release()  # release previous video writer
                      if vid_cap:  # video
                          fps = vid_cap.get(cv2.CAP_PROP_FPS)
                          w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                          h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                      else:  # stream
                          fps, w, h = 30, im0.shape[1], im0.shape[0]
                          save_path += '.mp4'
                      vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                  vid_writer.write(im0)

  if save_txt or save_img:
      s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
      # print(f"Results saved to {save_dir}{s}")

  # print(f'Done. ({time.time() - t0:.3f}s)')
def get_iou(pred_box, gt_box):
  """
  pred_box : the coordinate for predict bounding box
  gt_box :   the coordinate for ground truth bounding box
  return :   the iou score
  the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
  the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
  """
  # 1.get the coordinate of inters
  ixmin = max(pred_box[0], gt_box[0])
  ixmax = min(pred_box[2], gt_box[2])
  iymin = max(pred_box[1], gt_box[1])
  iymax = min(pred_box[3], gt_box[3])

  iw = np.maximum(ixmax-ixmin+1., 0.)
  ih = np.maximum(iymax-iymin+1., 0.)

  # 2. calculate the area of inters
  inters = iw*ih

  # 3. calculate the area of union
  uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
          (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
          inters)

  # 4. calculate the overlaps between pred_box and gt_box
  iou = inters / uni

  return iou
#MAIN_________________________________

craft_net,craft_refine_net,craft_args=load_detection_model()
recognizer=load_recognition_model()
opt,model,stride=load_yolo()
fontpath = "./arial.ttf" # <== download font
font = ImageFont.truetype(fontpath, 20)
def sepia(videoFile):
  # videoFile=input("Input video_path (type 'ESC' to exit!) :")

  #get 1 frame per second
  # videoFile = "/content/LoLstream_test2.mp4"
  # imagesFolder = "./frames"
  # if not os.path.isdir(imagesFolder):
  #   os.mkdir(imagesFolder)

  cap = cv2.VideoCapture(videoFile)
  frameRate = cap.get(5) #frame rate
  width  = int(cap.get(3))  # float `width`
  height = int(cap.get(4))
  size=(width, height)
  image_name="0"
  fps=20
  # cv2.VideoWriter_fourcc(*'MP4V')
  writer = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
  # writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, size)
  while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
      break
    
    if (frameId % math.floor(frameRate) == 0):
      # imgtest = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      imgtest=frame
      # imgtest=Image.fromarray(frame)
      # imgtest = imgtest[:,:,::-1]
      # imgtest.save("./temp1.jpg")
      image_name=str(int(frameId))
      print('frame id: ',image_name)
      # if not os.path.isdir('./temp'):
      #   os.mkdir('./temp')
      # cv2.imwrite("./temp/"+image_name+".jpg", imgtest)

      infer_detection(imgtest,image_name,craft_net,craft_refine_net,craft_args)
      crop_OCR(recognizer, imgtest,image_name)

      # minimap=imgtest.crop((1625, 785, 1920, 1080))
      minimap = imgtest[785:1080, 1625:1920]
      cv2.imwrite("./temp.jpg", minimap)
      # minimap.save("./temp.jpg")
      opt.source="./temp.jpg"
      if os.path.exists("./runs/detect/exp/labels/temp.txt"):
        os.remove("./runs/detect/exp/labels/temp.txt")
      detect(opt,model,stride,save_img=False)
      fps=20
    #visualize output frame: OCR
    if (fps>0):
      imgtest=frame
      data=json.load(open('./Results/data.json'))
      for k,v in data[image_name].items():
        bbox= v['box']
        bbox = np.array([[int(float(i)) for i in coord] for coord in bbox])
        # print(k)
        try:
          label= v['trans']
        except:
          label=' '
        bbox_= bbox.reshape((-1,1,2))
        cv2.polylines(imgtest,[bbox_],True,(0,0,255))
        imgtest=Image.fromarray(imgtest)
        draw = ImageDraw.Draw(imgtest)
        draw.text((bbox[0][0],bbox[0][1]),label, font = font,fill=(0,0,255,255))
        imgtest = np.array(imgtest)
      #visualize output frame: YOLO
      if os.path.exists("./runs/detect/exp/labels/temp.txt"):
        data=open("./runs/detect/exp/labels/temp.txt",'r')

        boxes=[]
        for i in data:
          x=i.rstrip().split(' ')[1:5]
          boxes.append([int(x[0])+1625, int(x[1])+785, int(x[2])+1625, int(x[3])+785])
        res = np.zeros(len(boxes),dtype=int)
        color=0
        final={}
        for i in range (len(boxes)-1):
          if(res[i]==0):
            color+=1
            res[i]=color
            final[str(res[i])]=[boxes[i]]
          for j in range (i+1,len(boxes)):
            iou=get_iou(boxes[i],boxes[j])
            if (iou>=0.1 and res[i]!=res[j]):
              res[j]=res[i]
              final[str(res[i])].append(boxes[j])
        if (str(res[-1]) in final):
          final[str(res[-1])].append(boxes[-1])
        else:
          final[str(res[-1])]=[boxes[-1]]
        for k,v in final.items():
          if(len(v)>1):
            pts=[]
            v=np.array(v)
            # c1=(int(np.mean(v[:,0])),int(np.mean(v[:,1])))
            # c2=(int(np.mean(v[:,2])),int(np.mean(v[:,3])))
            x_cen=int((int(np.mean(v[:,0]))+int(np.mean(v[:,2])))/2)
            y_cen=int((int(np.mean(v[:,1]))+int(np.mean(v[:,3])))/2)
            # cv2.rectangle(imgtest, c1, c2, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(imgtest, '!!!', (x_cen,y_cen), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
        for x in boxes:
          c1, c2 = (x[0], x[1]), (x[2], x[3])
          cv2.rectangle(imgtest, c1, c2, (0,255,255), thickness=1, lineType=cv2.LINE_AA)
      writer.write(imgtest)
      fps-=1
  if os.path.exists("./temp.jpg"):
    os.remove("./temp.jpg")
  if os.path.exists("./runs"):
    shutil.rmtree("./runs")
    

  cap.release()
  writer.release()
  print("Done!")
  return "./output.mp4"

iface = gr.Interface(fn=sepia,inputs=gr.inputs.Video(label="Input Video"),outputs=gr.outputs.Video(label="Output Video"),interpretation="default")
# iface = gr.Interface(fn=sepia,inputs=gr.inputs.Image(),outputs="text")
iface.launch(debug=False,share=True)
input("Running....")
