# USAGE
#python test.py --trained_model=craft_mlt_25k.pth --test_folder=./test

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from math import *
from craft import CRAFT
import pytesseract
import time
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import imutils





from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/test/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def rotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
    #cv2.imshow("imgOut",imgOut)  #裁减得到的旋转矩形框
    
    #cv2.imwrite("imgOut.jpg",imgOut)
    # pt2 = list(pt2)
    # pt4 = list(pt4)
    # [[pt2[0]], [pt2[1]]] = np.dot(matRotation, np.array([[pt2[0]], [pt2[1]], [1]]))
    # [[pt4[0]], [pt4[1]]] = np.dot(matRotation, np.array([[pt4[0]], [pt4[1]], [1]]))
    # pt1 = (int(pt1[0]), int(pt1[1]))
    # pt2 = (int(pt2[0]), int(pt2[1]))
    # pt3 = (int(pt3[0]), int(pt3[1]))
    # pt4 = (int(pt4[0]), int(pt4[1]))
    # drawRect(imgRotation,pt1,pt2,pt3,pt4,(255,0,0),2)
    return imgOut

def measure_blurr(img):
    img_graya = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_gray = cv2.GaussianBlur(img_graya, (3,3), 0)
    return cv2.Laplacian(img_gray,cv2.CV_64F).var()
    
    
def compute_diff(prvimg, curimg):
    prv_greya = cv2.cvtColor(prvimg,cv2.COLOR_RGB2GRAY)
    cur_greya = cv2.cvtColor(curimg,cv2.COLOR_RGB2GRAY)
    prv_grey = cv2.GaussianBlur(prv_greya, (19,19), 0)
    cur_grey = cv2.GaussianBlur(cur_greya, (19,19), 0)
    frameDelta = cv2.absdiff(prv_grey, cur_grey)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    return sum(sum(thresh))
    #return thresh
    
if __name__ == '__main__':
    # load net
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
        args.poly = True

    (W, H) = (None, None)
    (newW, newH) = (320, 320)
    (rW, rH) = (None, None)
    
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    netss = cv2.dnn.readNet('frozen_east_text_detection.pb')
    
    cap = cv2.VideoCapture("tours.mp4")
    fr = cap.get(cv2.CAP_PROP_FPS)
    print(fr)
    #cap = cv2.VideoCapture(0)
    ## some videowriter props
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 3
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    vout = cv2.VideoWriter()
    vout.open('output.mp4',fourcc,fps,sz,True)
    ret, firstFrame = cap.read()
    score_queue = []
    score_product = 0
    ea = 0
    cr = 0
    sk = 0
    t = time.time()
    while(1):
        lx = int((time.time() - t) * fr)
        cap.set(cv2.CAP_PROP_POS_FRAMES,lx)
        ret, frame = cap.read()
        
        bflag = 0
        mflag = 0
        mode = 0
        
        bm = measure_blurr(frame)
        #print('measure_blurr' + str(bm))
        
        mm = compute_diff(firstFrame, frame)
        firstFrame = copy.deepcopy(frame)
        #time.sleep(0.3)
        
        if (bm < score_product * 0.85 / 25) or (mm > 190000):
            bflag = 1
        if mm > 150000:
            mflag = 1
        
        score_queue.append(bm)
        score_product += bm
        if len(score_queue) > 25:
            cc = score_queue.pop(0)
            score_product -= cc
        
        if bflag == 0 and mflag == 0:
            mode = 1
            bboxes, polys, score_text = test_net(net, frame, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
            for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape((-1))
                
                pt1 = [poly[0], poly[1]]
                pt2 = [poly[2], poly[3]]
                pt3 = [poly[4], poly[5]]
                pt4 = [poly[6], poly[7]]
                ptm = [(poly[0]+poly[2]+poly[4]+poly[6]) / 4, (poly[1]+poly[3]+poly[5]+poly[7]) / 4]
                
                cp =  copy.deepcopy(frame)
                zz = rotateImage(cp,-degrees(atan2(pt1[1] - pt2[1],pt2[0]-pt1[0])),pt1,pt2,pt3,pt4)
                text=pytesseract.image_to_string(zz)
                cv2.putText(frame,text, (poly[0],poly[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,255,10), 1, cv2.LINE_AA)
                #strResult = ','.join([str(p) for p in poly]) + '\r\n'
                #print(text)
                
                poly = poly.reshape(-1, 2)
                cv2.polylines(frame, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                
        if (bflag == 1 and mflag == 0) or (bflag == 0 and mflag == 1):
            mode = 2
            rezframe = copy.deepcopy(frame)
            orig = copy.deepcopy(frame)
            
            if W is None or H is None:
                (H, W) = rezframe.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)
            
            rezframe = cv2.resize(rezframe, (newW, newH))
            blob = cv2.dnn.blobFromImage(rezframe, 1.0, (newW, newH),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
            netss.setInput(blob)
            (scores, geometry) = netss.forward(layerNames)
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            results = []
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                roi = orig[startY:endY, startX:endX]
                config = ("-l eng --oem 1 --psm 7")
                try:
                    text = pytesseract.image_to_string(roi, config=config)
                    results.append((startX, startY, endX, endY))
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame,text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,10,255), 1, cv2.LINE_AA)
                except:
                    pass
            
        if bflag == 1 and mflag == 1:
            mode = 3
        
        cv2.putText(frame,'blurr: ' + str(bm), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,255,10), 1, cv2.LINE_AA)
        cv2.putText(frame,'motion: ' + str(mm), (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,255,10), 1, cv2.LINE_AA)
        if mode == 1:
            cr+=1
            cv2.putText(frame,'mode: CRAFT', (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(250,10,10), 1, cv2.LINE_AA)
        if mode == 2:
            ea+=1
            cv2.putText(frame,'mode: EAST', (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,250,10), 1, cv2.LINE_AA)
        if mode == 3 or mode == 0:
            sk+=1
            cv2.putText(frame,'mode: SKIP', (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(10,10,250), 1, cv2.LINE_AA)
        
        cv2.imshow("capture", frame)
        vout.write(frame)
        #print('time  '+str(time.time() - t))
        print('SKIP: ' + str(sk) + '  CRAFT: ' + str(cr) + '  EAST: ' + str(ea))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    vout.release()
    cv2.destroyAllWindows()
    
    #python test.py --trained_model=craft_mlt_25k.pth --test_folder=./test