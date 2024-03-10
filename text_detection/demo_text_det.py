"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
import torch
# import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import text_detection.craft_utils as craft_utils
import text_detection.imgproc as imgproc
import text_detection.file_utils as file_utils
from text_detection.craft import CRAFT

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

def test_net(args, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
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

def crop_and_save_images(filename, image, frame_idx, polys, save_folder):

    # Create a folder for saving cropped images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Iterate through bounding boxes and crop regions
    for i, points in enumerate(polys):
        # Reshape points to get x, y coordinates separately
        xy_coordinates = np.array(points).reshape(-1, 2)

        # Find minimum and maximum x, y coordinates
        x_min, x_max = np.min(xy_coordinates[:, 0]), np.max(xy_coordinates[:, 0])
        y_min, y_max = np.min(xy_coordinates[:, 1]), np.max(xy_coordinates[:, 1])

        # Convert coordinates to integers and ensure they're within image bounds
        x_min, x_max = int(max(0, x_min)), int(min(image.shape[1], x_max))
        y_min, y_max = int(max(0, y_min)), int(min(image.shape[0], y_max))

        # Crop the region from the original image based on the bounding box
        cropped_region = image[y_min:y_max, x_min:x_max]

        # Save the cropped region as a new image in the specified folder
        save_path = os.path.join(save_folder, f"{filename}_crop_{i}_frame_{frame_idx}.jpg")
        cv2.imwrite(save_path, cropped_region)


def text_detection(args):

    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.det_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.det_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.det_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    online_video = args.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    """ For test images in a folder """
    # Set Dataloader
    if online_video:
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = file_utils.LoadStreams(args.source)
    else:
        dataset = file_utils.LoadImagesVideos(args.source)

    t = time.time()
    # load data
    input_video_fps = 0
    for frame_idx, (image_path, image, fps) in enumerate(dataset):
        print("batch_size_det", torch.from_numpy(image).size(0))
        if online_video:  # batch_size >= 1
            image_path = Path(image_path[frame_idx])

        # print("Test image {:d}/{:d}: {:s}".format(frame_idx + 1, len(dataset), image_path), end='\r')
        bboxes, polys, score_text = test_net(args, net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net)

        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # save cropped images around text detections
        crop_and_save_images(filename, image, frame_idx, polys, args.cropped_images_folder)
        input_video_fps = fps
        # save score text
        # file_utils.saveResult(filename, frame_idx, image, polys, dirname=args.result_folder)

    # print("Elapsed time for detections and saving: {}s".format(time.time() - t))
    return input_video_fps
