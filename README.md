# ICRA-40-Dataset
## Ways of handling the dataset
### Data Collection
We use the save_dataset_show.py script to collect data. The script uses asynchronous multithreading for processing, and has a higher frequency of accelerometers than other data sets recorded by Frodobots. In the case of the images, since we don't have access to the robot server, we only get a frequency of 3 Hz. The other frequencies are consistent with the other datasets recorded by Frodobots.
### Face Blur
We use the Yunet(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) as the face detector and use the Gaussian blur for every face. The detail is shown in face_blur_light.py.
### Depth prediction
We use the SOTA real-time Monodepth estimator-LiteMono to predict the depth from the monocular camera. We have 3 different types of depth file, the npy file, the depth image and the hotmap image.
We run the following code to do the depth prediction.
""bash
python pic_depth_prediction.py     --input_folder /media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/rear     --output_folder /media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/rear_depth     --load_weights_folder /home/zhangqi/Documents/Library/Lite-Mono/pretrained_model     --model lite-mono-8m
""
