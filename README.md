# ICRA-40-Dataset
For the details please refer to https://staff.fnwi.uva.nl/a.visser/publications/zhang2024frodobots.pdf.
## Ways of handling the dataset
### Data Collection
We use the save_dataset_show.py script to collect data. The script uses asynchronous multithreading for processing, and has a higher frequency of accelerometers than other data sets recorded by Frodobots. In the case of the images, since we don't have access to the robot server, we only get a frequency of 3 Hz. The other frequencies are consistent with the other datasets recorded by Frodobots.
### Face Blur
We use the Yunet(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) as the face detector and use the Gaussian blur for every face. The detail is shown in face_blur_light.py.
### Depth Prediction
We use the SOTA real-time Monodepth estimator-LiteMono to predict the depth from the monocular camera. We have 3 different types of depth file, the npy file, the depth image and the hotmap image.
We run the following code to do the depth prediction.
""bash
python pic_depth_prediction.py     --input_folder /media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/rear     --output_folder /media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/rear_depth     --load_weights_folder /home/zhangqi/Documents/Library/Lite-Mono/pretrained_model     --model lite-mono-8m
""
## How To Download The Dataset
We store our dataset in the Uva server
https://uvaauas.figshare.com/articles/dataset/frodobot_recording_2024_Sep_25_19_14_19_59_zip/27127125?file=49473699

## How To Use The Dataset
Although we have a format different from other Frodobots datasets, we can also use the same method with small modifications. So, you can refer my other rep to use ORBSLAM3, YOLOX and LiteMono on it.
https://github.com/SlamMate/vSLAM-on-FrodoBots-2K

PS, control_log.csv is the data of the human manipulating data. U can use the vision image and human operating to train the controlling method of small car.

## Citation
To cite this work, please use the following reference in English:

```plaintext
@misc{zhang2024earthroverdatasetrecorded,  
      title={An Earth Rover dataset recorded at the ICRA@40 party},  
      author={Qi Zhang and Zhihao Lin and Arnoud Visser},  
      year={2024},  
      eprint={2407.05735},  
      archivePrefix={arXiv},  
      primaryClass={cs.RO},  
      url={https://arxiv.org/abs/2407.05735}  
}
```
