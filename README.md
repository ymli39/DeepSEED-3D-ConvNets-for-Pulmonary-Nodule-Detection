# DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection
DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder ConvNets for Pulmonary Nodule Detection

-------------------------------------------------------------
Major update in 2023 Feb 17

Thank you all for your support to this project and appreciated all your contributions. 

I have updated the evaluation script for LNUA16 data that the previous code contains a bug during evaluation.

please check the file "noduleCADEvaluationLUNA16.py" if you downloaded my code prior to 2023 Feb 17.

I have added the test file under directory: ./luna_detector/labels/luna_test.csv

you can directly run file noduleCADEvaluationLUNA16.pyto check the results. The example results are generated in folder: /luna_detector/test_results/predanno0.3.csv for test fold 9 (might be different in your case).

Bug is caused by ID mismatch, for example ID 56, in predicted file the script reads the id as '56' but in gt file it reads id as '056'. You might not encounter this issue with other benchmark code since they kept original dicom id as the file id name (i.e., xxx.xxxxxx.xxxxxxxxx). 

During preprocessing, I renamed all files starts from 0 to 888. The bug apears for any ID numbers ranged from 0 to 99.

-------------------------------------------------------------

Dataset:
LUNA16 can be downloaded from https://luna16.grand-challenge.org/data/

LIDC-IDR can be downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI




-------------------------------------------------------------
Preprocessing:
Go to config_training.py, create two directory Luna_data and Preprocess_result_path. Then change directory listed as follows:

Luna_raw: raw data folder downloaded from LUNA16 website

Luna_segment: luna segmentation download from LUNA16 website

Luna_data: temporary folder to store luna data

Preprocess_result_path: final preprocessed data folder

Run prepare.py, output LUNA16 data can be found inside folder Preprocess_result_path, with saved images as _clean.npy, _label.npy for training, and _spacing.npy, _extendbox.npy, _origin.npy are separate information for data testing.



-------------------------------------------------------------
Training:
Go to ./detector directory, the model can be trained by calling the following script:

	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_detector_se.py -b 16 --save-dir /train_result/ --epochs 150

The output model can be found inside ./train_result/ folder.



-------------------------------------------------------------
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

---------------------------------------------------------------

You could refer to the arxiv paper for more details and performance:

	@inproceedings{li2020deepseed,
	  title={DeepSEED: 3D Squeeze-and-Excitation Encoder-Decoder Convolutional Neural Networks for Pulmonary Nodule Detection},
	  author={Li, Yuemeng and Fan, Yang},
	  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
	  pages={1866--1869},
	  year={2020},
	  organization={IEEE}
	}
