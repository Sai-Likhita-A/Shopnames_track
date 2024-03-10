# Import default libraries

import string
import argparse
from text_detection.demo_text_det import *
from text_recognition.demo_ocr import *

if __name__ == '__main__':
    # arguments for text detection from video frames
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--det_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.4, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
    parser.add_argument('--source', default=' https://www.youtube.com/watch?v=Xe3evWq5tas', type=str, help='path to input images or videos')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                        help='pretrained refiner model')
    parser.add_argument('--result_folder', default='results2',
                        help='path to image_folder which contains cropped text images')
    parser.add_argument('--cropped_images_folder', default='results2/cropped_images_folder',
                        help='path to image_folder which contains cropped text images')
########################################################################################################################
    # arguments for text recognition from text detected video frames
    parser.add_argument('--text_rec_model', default='weights/TPS-ResNet-BiLSTM-Attn.pth',
                        help="path to saved_model to evaluation")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--filename', default='shopnames_timestamp.csv',
                        help='path to excel file to write results')

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', default=False, action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()

    # creating a directory to store results if not already exists
    if not os.path.isdir(args.result_folder):
        os.mkdir(args.result_folder)

    t = time.time()
    input_video_fps = text_detection(args)
    # input_video_fps = 25.0
    print("Elapsed time for text detection : {}s".format(time.time() - t))
    print('-' * 70)

    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()
    t = time.time()
    demo(args, input_video_fps)
    print("Elapsed time for ocr: {}s".format(time.time() - t))
