"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-

import os
import time

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

# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--det_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default=r'D:\Pycharm Projects\Version_3_text_detection_OCR\test_images', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
#
# args = parser.parse_args()



def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5) #  args.canvas_size , args.mag_ratio
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

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def crop_and_save_images(image_path, polys, save_folder):
    # Load the original image
    image = cv2.imread(image_path)

    # Create a folder for saving cropped images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename, file_ext = os.path.splitext(os.path.basename(image_path))
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
        save_path = os.path.join(save_folder, f"{filename}_crop_{i}{file_ext}")
        # save_path = os.path.join(save_folder, f"cropped_image_{i}_{os.path.basename(image_path)}")
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
    # if args.refine:
    #     from refinenet import RefineNet
    #     refine_net = RefineNet()
    #     print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    #     if args.cuda:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
    #         refine_net = refine_net.cuda()
    #         refine_net = torch.nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
    #
    #     refine_net.eval()
    #     args.poly = True

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    # result_folder = './result/'
    # if not os.path.isdir(result_folder):
    #     os.mkdir(result_folder)

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        # print("image shape after load", np.shape(image))
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net)

        # save cropped images around text detections
        crop_and_save_images(image_path, polys, args.cropped_images_folder)
        # print("Polys:", polys)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = args.result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=args.result_folder)

    print("Elapsed time for detections and saving: {}s".format(time.time() - t))

# if __name__ == '__main__':
#     text_detection(args)