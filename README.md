# DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection
DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder ConvNets for Pulmonary Nodule Detection

Dataset:
LUNA16 can be downloaded from https://luna16.grand-challenge.org/data/

LIDC-IDR can be downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI


Preprocessing:
Go to config_training.py, create two directory Luna_data and Preprocess_result_path. Then change directory listed as follows:
Luna_raw: raw data folder downloaded from LUNA16 website
Luna_segment: luna segmentation download from LUNA16 website
Luna_data: temporary folder to store luna data
Preprocess_result_path: final preprocessed data folder

Run prepare.py, output LUNA16 data can be found inside folder Preprocess_result_path, with saved images as _clean.npy, _label.npy for training, and _spacing.npy, _extendbox.npy, _origin.npy are separate information for data testing.

Training:
Go to ./detector directory, the model can be trained by calling the following script:

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_detector_se.py -b 16 --save-dir /train_result/ --epochs 150

The output model can be found inside ./train_result/ folder.

Testing:
In order to obtain the predicted label, go to ./detector directory, the model can be tested by calling the following script:

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_detector_se.py -b 1 --resume ‘best_model.ckpt’ --test 1 --save-dir /output/

The bbox output can be found inside ./output/bbox/, the predicted bounding boxes and ground truth bounding boxes will be saved in this direction.

Then for FROC metric evaluation, go to FROCeval.py, change path for following directories:
seriesuids_filename: patient ID for testing
detp: threshold for bounding boxes regression
nmsthresh: threshold used bounding boxes non-maximum suppression
bboxpath: directory which stores bounding boxes from testing output
Frocpath: path to store FROC metrics
Outputdir: store the metric evaluation output

The FROC evaluation script is provided from LUNA16 website, you can find the script in noduleCADEvaluationLUNA16.py. 
