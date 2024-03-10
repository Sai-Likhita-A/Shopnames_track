import time
import torch
import os
import torch.utils.data
import torch.nn.functional as F
import csv
import datetime

from text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recognition.dataset import RawDataset, AlignCollate
from text_recognition.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print("device", device)
def demo(opt, input_video_fps):
    """ model configuration """

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


    # Field names of the results to write to Excel file
    fields = ['Shop_name', 'Timestamp_of_first_frame', 'Timestamp_of_last_frame']
    # predict
    model.eval()
    all_text_predicted=[]
    time_stamp = []
    final_text_predicted = []
    # t = time.time()
    with torch.no_grad():

        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            # print("batch_size_det", batch_size)
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
            # t1 = time.time()
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                filename, file_ext = os.path.splitext(os.path.basename(img_name))
                seperated_list = filename.split("_")
                frame_no = int(seperated_list[-1])
                timestamp_sec = frame_no/input_video_fps

                timestamp_hrs = datetime.timedelta(seconds=timestamp_sec)
                # print("frame_no, timestamp_hrs:", frame_no, timestamp_hrs)
                all_text_predicted.append(pred)
                time_stamp.append(timestamp_hrs)
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

    length_for_pred_text = len(all_text_predicted)
    unique_list = (list(set(all_text_predicted)))
    # print("all_text_predicted: ", all_text_predicted , "\ntime_stamp:", time_stamp)
    for shop_name in unique_list:
        indexes = [index for index in range(length_for_pred_text) if all_text_predicted[index] == shop_name]
        start_t = min(indexes)
        stop_t = max(indexes)
        # Data to be written to the CSV file
        final_text_predicted.append([shop_name, time_stamp[start_t], time_stamp[stop_t]])
        # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')


    # Write data to the CSV file with a pipe delimiter
    with open(opt.filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(final_text_predicted)
    print(f"Data has been written to {opt.filename}")
