# FaceDetectionWithName

## Today’s blog post is broken down into three parts.

In the first part we’ll discuss the origin of the more accurate OpenCV face detectors and where they live inside the OpenCV library.

From there I’ll demonstrate how you can perform face detection in images using OpenCV and deep learning.

I’ll then wrap up the blog post discussing how you can apply face detection to video streams using OpenCV and deep learning as well.


###### Our project has four directories in the root folder:

1. dataset/ : Contains our face images organized into subfolders by name.

2. images/ : Contains three test images that we’ll use to verify the operation of our model.

3. face_detection_model/ : Contains a pre-trained Caffe deep learning model provided by OpenCV to detect faces. This model detects and localizes faces in an image.

4. output/ : Contains my output pickle files. If you’re working with your own dataset, you can store your output files here as well. The output files include:
	- embeddings.pickle : A serialized facial embeddings file. Embeddings have been computed for every face in the dataset and are 				      stored in this file.
	- le.pickle : Our label encoder. Contains the name labels for the people that our model can recognize.
	- recognizer.pickle : Our Linear Support Vector Machine (SVM) model. This is a machine learning model rather than a deep learning model and it is responsible for actually recognizing faces.

##Commands to execute the project: 

######Extract embedding
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

######Training the model
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

######Recognize faces with opencv
python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image images/kiran.jpg
	
***********************************************Recognize faces with opencv in video stream********************
python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
 
