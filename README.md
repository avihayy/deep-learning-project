# Traffic congestion detection using YOLOv3 #

In Our study we used computer vision and deep learning in order to detect and classify traffic congestion from camera images. We based on YOLOv3[15] (you only look once), which is currently the state of the art for real time processing method.
We have focused especially at crossroads, since there the high congestion can be prevented by proper regulation of traffic and the utilization of smart traffic lights.
Our Network is trained to detect all vehicles in each frame. We used videos of car traffic at different times and at different traffic congestion in order to show that our model can perform well in challenging conditions as well. 

#### This repository contains the following files: ####

Detection folder:
* Detector.py
*	model.py
*	yolo.py
*	yolo_anchors.txt
*	data_classes.txt

Training folder:
*	Train_Main.py
*	Train_Model.py
*	Train_Utils.py
*	csv2txt_convert.py
*	yolo_anchors.txt
*	data_classes.txt

Since our Data-set is very big, we didn't able to upload it to the github, therefore you can find our data-set and the pre-trained weights at the following Drive link:
https://drive.google.com/drive/folders/1rHVssHFWw_SHX45BQcnHLU6HIFm_HIRk?usp=sharing

#### How to run: ####

a.	Convert the annotation file from csv file to txt file:
convert the annotation file from csv file to txt file by using the module “csv2txt_convert.py” and insert it the following switches:
1.	--Frames_Folder: Absolute path to the folder of the csv file and the tagged frames. 

The csv file with the tagged frames are in the folder we shared in the drive link, “Data”, at the path:

Data/Source_Video/Training_Frames/location_name

And the location-name can be one of the following:
* HATZAV
* Jisr
* Kfar_saba_east
* Ofakim
* Holot

2.	--camera_name - the name of the camera (location), should be match to the name of the location that the frames were taken.

In our case, Can be one of the following:
* HATZAV : "HATZAV"
* Jisr: "jisr"
* Kfar saba east: "kfar"
* Ofakim: "ofakim"
* Holot: "holot".



b.	Training Our model
Train our model using the module “Train_Main.py” and insert it the following switches:
1.	– epochs - choose the number of epochs for the training, defult is -300
2.	--annotation_file - Absolute Path to the annotations file: “data_train.txt “ that you made with csv2txt_convert.py.
3.	-- weights_folder_path - choose the absolute path where you want the weights will be stored in. If None weights won't be saved.
4.	-- pre_trained_path - Absolute path for pre trained weights.
If None the original weights of COCO will be loaded.

All our pre-trained weights are in the folder we shared in the drive link, “Data”, at the path: 
Data/Model_Weights/location_name/trained_weights_final.h5"
where location name is: HATZAV/jisr/kfar_saba_east/ofakim/holot, according to your choice (must be consistent in all stages).

c.	Testing Our Model:
Test our model by using the module “Detector.py” and insert it the following switches:
1.	--threshold_mode - set the threshold mode. Can be one of the following:
* "counting" = threshold define by number of the vehicles (integer) in threshold_low, threshold_high.
* "density" = threshold define by the density of the vehicles (float between 0-1) in threshold_low, threshold_high.
* "velocity" = threshold define by the velocity of the vehicles (integer which indicates [kmh]) in threshold_low, threshold_high.
2.	--threshold_low - set the upper threshold density for low congestion.
3.	--threshold_high - set the lower threshold density for high congestion.
4.	--input_path - Absolute path to video directory for making detection.
All the videos for tests are in the folder we shared in the drive link, “Data”, at the path:
Data/Source_Video/Video_Test/location name"
where location name is: HATZAV/jisr/kfar_saba_east/ofakim/holot 
(must be consistent in all stages).
5.	--output_path - Output path for detection video results.

