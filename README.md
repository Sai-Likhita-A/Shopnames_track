## Getting Started

### Overview
PyTorch implementation of a system that extracts shop names from a video featuring a shopping mall or a marketplace. To detect shop names within the video frames CRAFT text detector that detect text area is used. The frames are cropped around the bounding box of texts detected in the first stage and are given as input to text recognition model. PyTorch implementation of four-stage STR framework in 
deep-text-recognition-benchmark model is used for OCR. The text output of the recognition model is compiled into an Excel spreadsheet with three columns: 'Shop Name', 'Timestamp_of_first_frame' and 'Timestamp_of_last_frame' of the shop appearance in the video.

### Install dependencies
#### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- check requiremtns.txt
```
pip install -r requirements.txt
```

### Test instruction using pretrained model
- Download the pre-trained weights of text detection model from [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) and place in weights folder
- Download pretrained weights of OCR model TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn.pth) from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) and place in weights folder
- Download pretrained weights of case sensitive version of OCR model TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn-case-sensitive.pth) from [here](https://drive.google.com/file/d/1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY/view) and place in weights folder

Run demo.py (add `--sensitive` option if you use case-sensitive model)
```
CUDA_VISIBLE_DEVICES=0 python3 infer.py \
--det_model 'weights/craft_mlt_25k.pth' --source test_images --result_folder 'results2' --cropped_images_folder 'results2/cropped_images_folder' \
--text_rec_model 'weights/TPS-ResNet-BiLSTM-Attn.pth' --filename 'shopnames_timestamp.csv' --Transformation TPS --FeatureExtraction ResNet \
--SequenceModeling BiLSTM --Prediction Attn
```

The resultant cropped images will be saved to `./result_folder/cropped_images_folder` by default
The text output is saved to shopnames_timestamp.csv along with the time stamps of its appearance in the input video

### Arguments
* `--det_model`: pretrained text detection model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--cuda`: use cuda for inference (default:True)
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--source`: path to input images or videos
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model
* '--result_folder': path to image_folder to save detection result
* '--cropped_images_folder': path to save cropped images around detected text
* `--text_rec_model`: pretrained OCR model
* `--workers`: number of data loading workers
* `--batch_size`: input batch size
* `--filename`: path to excel file to write results
* `--imgH`: the height of the input image
* `--imgW`: the width of the input image
* `--rgb`: the width of the input image
* `--character`: character label (default='0123456789abcdefghijklmnopqrstuvwxyz')
* `--sensitive`: for sensitive character mode
* `--PAD`: whether to keep ratio then pad for image resize
* `--Transformation`: select Transformation module [None | TPS]
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet]
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM]
* `--Prediction`: select Prediction module [CTC | Attn]
* `--num_fiducial`: number of fiducial points of TPS-STN
* `--input_channel`: the number of input channel of Feature extractor
* `--output_channel`: the number of output channel of Feature extractor
* `--hidden_size`: the size of the LSTM hidden state
    
Acknowledgements
This implementation has been based on these repository https://github.com/clovaai/CRAFT-pytorch, https://github.com/clovaai/deep-text-recognition-benchmark.
