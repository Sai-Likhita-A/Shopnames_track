import time
import torch
# import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recognition.dataset import RawDataset, AlignCollate
from text_recognition.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    # print("opt: ", opt)
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.text_rec_model)
    model.load_state_dict(torch.load(opt.text_rec_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.cropped_images_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    log = open(f'./log_demo_result.txt', 'a')
    dashed_line = '-' * 80
    head = f'{"image_path":25s}\t\t{"predicted_labels":25s}\tconfidence score'
    head1 = f'{"shop_name":25s}\t{"start_time":25s}\tend_time'
    # print(f'{dashed_line}\n{head}\n{dashed_line}')
    # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

    # predict
    model.eval()
    all_text_predicted=[]
    time_stamp = []
    final_text_predicted = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            t = time.time()

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                # print(time.time())
                t1 = time.time()-t
                # print(t1)
                all_text_predicted.append(pred)
                time_stamp.append(t1)
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

    length_for_pred_text = len(all_text_predicted)
    unique_list = (list(set(all_text_predicted)))
    log.write(f'{dashed_line}\n{head1}\n{dashed_line}\n')
    for c in unique_list:
        indexes = [index for index in range(length_for_pred_text) if all_text_predicted[index] == c]
        start_t = min(indexes)
        stop_t = max(indexes)
        final_text_predicted.append([c, time_stamp[start_t], time_stamp[stop_t]])
        # log.write(f'{c:25s}\t{time_stamp[start_t]:0.4f}\t{time_stamp[stop_t]:0.4f}\n')
    print("before",final_text_predicted)
    text=sorted(final_text_predicted)

    # print(f'{all_text_predicted}')
    print("after",text)
    # log.write(f'{final_text_predicted}')
    log.close()

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', default=r'D:\Pycharm Projects\Version_3_text_detection_OCR\test_images', help='path to image_folder which contains text images')  # 'demo_image' r'D:\Pycharm Projects\Version_3_text_detection_OCR\test_images'
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    # parser.add_argument('--text_rec_model', default='weights/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
    # """ Data processing """
    # parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    # parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    # parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    # parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    # parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    # """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    # parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    # parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    # parser.add_argument('--output_channel', type=int, default=512,
    #                     help='the number of output channel of Feature extractor')
    # parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    #
    # opt = parser.parse_args()

    # """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    #
    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()
    # t = time.time()
    # demo(opt)
    # print("Total elapsed time : {}s".format(time.time() - t))
