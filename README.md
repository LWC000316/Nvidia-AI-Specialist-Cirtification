![NVIDIA LOGO](https://github.com/user-attachments/assets/9cf87f01-ff75-4c6a-b4c8-2560ca2e4db7)

# Nvidia AI Specialist Certification
### <span style="color:violet">Title : ADAS TSR(Traffic Sign Recognition) using YOLOv5</span>
---
## ✅ OverView of the Project
    - Opening background information

    - General description of the current project

    - Proposed idea for enhancements to the project

    - Value and significance of this project

    - Current limitations

    - Literature review
---
## Opening background information
```
With the recent development of autonomous driving technology, the importance of ADAS (Advanced Driver Assistance Systems),
which is a driver assistance system, is increasing.
In particular, TSR (Traffic Sign Recognition) is a technology that helps drivers recognize road signs while driving,
ensuring they follow traffic regulations and are aware of important road conditions.
It is a very important part of the automobile industry.
```
---
## General description of the current project
```
This project aims to implement a Traffic Sign Recognition (TSR) system by utilizing YOLOv5, a deep learning technology.
We build a system that detects road signs in real time using YOLOv5 and provides warnings to drivers based on
this information to ensure they follow traffic regulations and avoid accidents. 
It responds to the recent development of deep learning technology and the increasing importance of ADAS systems,
and contributes to enhancing the safety and convenience of drivers.
```
---
## Proposed idea for enhancements to the project
```
High accuracy and real-time processing: YOLOv5 offers high accuracy and low processing latency,
enabling fast real-time road sign detection and recognition.
Flexibility and scalability: YOLOv5 has great flexibility to identify and track road signs
in various driving environments, making it easy to apply additional features and improvements.
```
---
## Value and significance of this project
```
The traffic sign recognition system helps drivers follow traffic regulations and provides critical road information
such as intersections, traffic lights, and roadworks.
The improved system through this project ensures driver safety and accelerates the development of autonomous driving technology.
By implementing TSR as part of ADAS, it enhances the driver experience and helps prevent traffic accidents.
```
---
## Current limitations
```
Current limitations include challenges in accurately recognizing road signs in diverse environments,
with performance potentially degraded by weather or lighting conditions.
Additionally, real-time processing and system optimization still need improvement.
```
---
## Literature review
```
In using YOLOv5 for this project, related papers are being investigated to consider the latest research and technology trends.
A literature review is being conducted on the performance and applicability of YOLOv5 for traffic sign recognition.
```
---
## <span style="color:blue"> Image Acquisition Method </span>
- I filmed it with a camera while driving on the highway.

[DRIVE TEST](https://github.com/user-attachments/assets/9bfaefa1-c508-4fa7-a04f-94441b3b1514)


## <span style="color:blue">Learning Data Extraction and Learning Annotation </span>

- In order to learn with 640 x 640 resolution images in YOLOv5,

  the images were first created as 640 x 640 resolution images.


## <span style="color:blue"> Video Resolution Adjustment </span>


<https://online-video-cutter.com/ko/resize-video>

![비디오 리사이저](https://github.com/user-attachments/assets/70e5e7a4-8d07-484c-bf78-bb652f0b381e)

- I used Darklabel to create edits with images based on frames, due to 640 x 640.
  

[DarkLabel2.4.zip](https://github.com/user-attachments/files/17794875/DarkLabel2.4.zip)

![darklabel 5](https://github.com/user-attachments/assets/1769e2b0-84ba-4854-beaa-2e4dd4cecf4c)

- First, add classes through darklabel.yml before annotation.


![다크라벨얌1](https://github.com/user-attachments/assets/3d8d6b52-dd4a-455e-9463-3f1f29f527de)

- Add vehicle classes in the yaml file and add vehicle license plate as the class name.

  
![다크라벨얌2](https://github.com/user-attachments/assets/b9811883-fc21-4f47-9985-ba8dca6c6fc6)

- When annotating, put the vehicle classes in classes_set so that you can see the classes set

  in the DarkLabel GUI, and set the name to be viewed in the GUI to vehicle license plate.
  

![다크라벨1](https://github.com/user-attachments/assets/cec14a11-6d59-4669-8883-472a8fe4925c)

- You can see that classes called vehicle classes have been added to the DarkLabel program,

  and a vehicle license plate has been added below.
  

![다크라벨2](https://github.com/user-attachments/assets/de621b99-0b97-4a89-aa57-160afbaea1bd)

- In the DarkLabel program, you can convert video into images frame by frame.

  First, select a 640 x 640 resolution video through Open Video. Afterwards,

  it is converted to an image through as images, and the labeled value is saved through GT save.

<img src="https://github.com/user-attachments/assets/0d4dac2b-8cab-4b5b-abca-67cc409a8c14" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/da7d2462-f52f-4a7c-88d6-53f2cf6cfaa3" width="50%" height="50%">
    
- You can see that labeled text documents and image files are in the labels folder and the images folder, respectively.

---

## NVIDIA JETSON NANO LEANING COURSE

- To install YOLOv5, clone the repository and install the packages specified in `requirements.txt`.

  Google Colaboratory was used and learning was conducted by linking to Google Drive.

  
```ipynb
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt
```

- Insert images and labeled values ​​into the images and labels folder in the Train folder to be trained.

<img src="https://github.com/user-attachments/assets/f95c88df-88cf-4d58-87cc-748065ae68e3" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/13e6cebd-11cd-4f21-abfa-d0e262cb428e" width="50%" height="50%">


- After preprocessing the image files in imagespath, save them as a single .npy file.

```ipynb
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

def _preproc(image, output_height=512, output_width=512, resize_side=512):
    ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

def Create_npy(imagespath, imgsize, ext) :
    images_list = [img_name for img_name in os.listdir(imagespath) if
                os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB")
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)
```

- Edit the `data.yaml` file to match the classes.

![데이터얌](https://github.com/user-attachments/assets/37464099-5da2-4959-a921-9887f8ae1799)


- Learning is conducted based on `data.yaml`.

```ipynb
!python train.py  --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5_2/yolov5/data.yaml --weights yolov5n.pt --cache
```


--`img 512` : This argument sets the image size to 512x512 pixels for training and inference. 

YOLOv5 models are trained on square images, and this parameter determines the resolution.

--`batch 16` : This specifies the batch size for training, meaning 16 images will be processed simultaneously in each iteration. 

Batch size can impact training speed and memory usage.

--`epochs 300` : This sets the number of training epochs to 300. An epoch represents one complete pass through the entire training dataset.

--`data /content/drive/MyDrive/yolov5/yolov5_2/data.yaml` : This argument points to the data.yaml file, which contains the configuration for your dataset, 

including the paths to your training and validation images and labels.

--`weights yolov5n.pt` : This specifies the initial weights to use for the model. 

yolov5n.pt represents a pre-trained YOLOv5 nano model, which can be used as a starting point for faster training.

--`cache` : This option enables caching of images to potentially speed up training, especially if you have a large dataset.


## learning results


- PR_Curve / F1_Curve

<img src="https://github.com/user-attachments/assets/18da92cc-ae8a-4dfe-b049-a858a8ea77f2" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/cde44001-7ed3-4705-9265-81b7c4f4fa6c" width="50%" height="50%">

- P_Curve / R_Curve

<img src="https://github.com/user-attachments/assets/80de5d47-00e1-4ba8-9360-1a5e01c5b4d5" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/50dc7fb6-7295-4080-9098-9994b6283792" width="50%" height="50%">

- confusion_matrix

<img src="https://github.com/user-attachments/assets/f4391b18-94f3-4242-a303-ce6673e526d3" width="50%" height="50%">

- labels / labels_correlogram

<img src="https://github.com/user-attachments/assets/4eb5e2a3-2930-4399-ad86-e09ef22d33aa" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/80cd0314-f7cc-4482-b373-abbca8709a2e" width="50%" height="50%">


- results

![result](https://github.com/user-attachments/assets/7ccb32e7-3c38-4110-8d64-23400ccd28c8)


- val_batch1_pred / val_batch2_pred

<img src="https://github.com/user-attachments/assets/a1919dad-ab5d-48a6-a4a9-39dde76a5c12" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/4b30e42f-17a9-4ae6-a308-b62c7336a21f" width="50%" height="50%">

- learning file

    - [Learning Results Videos 1](https://drive.google.com/drive/folders/1mz9lRwu_nwsyweWgbLiDaz_5yb4ooJBt?dmr=1&ec=wgc-drive-hero-goto)

---

## detect results

- After completing training, run `detect.py` based on the image used for training.

```ipynb
!python detect.py --weights /content/drive/MyDrive/yolov5_2/yolov5/runs/train/exp5/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/yolov5_2/yolov5/Train/images
```


--`!python detect.py` : This part calls the Python interpreter to execute the detect.py script, 


which is responsible for running inference using a YOLOv5 model.

--`weights /content/drive/MyDrive/yolov5/yolov5/runs/train/exp5/weights/best.pt` : This argument specifies the path 

to the trained model weights file (best.pt).

This file contains the learned parameters of the model, allowing it to detect objects. 

It's likely that you trained the model in a previous step (exp5) and saved the best performing weights to this location.

--`img 512` : This argument sets the image size for inference to 512x512 pixels. 

This should match the image size used during training to ensure optimal performance.

--`conf 0.1` : This sets the confidence threshold for object detection. 

The model will only output detections with a confidence score of 0.1 or higher. 

This value can be adjusted to control the sensitivity of the detector. 

Lowering the confidence threshold will result in more detections, but may also increase the number of false positives.

--`source /content/drive/MyDrive/yolov5/yolov5/Train/images` : This argument specifies the path to the input 

images or directory of images that you want to run inference on. 

In this case, it's pointing to the Train/images directory, which likely contains the images you used for training. 

You can change this path to any directory containing images you want to analyze.




- Image produced through detect results

![00000266](https://github.com/user-attachments/assets/ce062078-9fda-4928-bac8-44230bb4bc7c)
![00000446](https://github.com/user-attachments/assets/c3a3379e-4332-4848-b0fa-83fc4e9e254e)
![00000612](https://github.com/user-attachments/assets/3ea49621-facf-41e4-9777-f9d78dacc795)


- Learning Results Images
 
    - <https://drive.google.com/drive/folders/1mRn3iXVgNcwv4lDLyZ3SSAaGnSZaqwNd?usp=drive_link>
 

- Video produced through detect results

<https://github.com/user-attachments/assets/c0a095ef-4a9b-4021-9d29-5df99beb701c)>


<https://github.com/user-attachments/assets/e71a996c-bf58-40c4-8ad9-13d49dc673e2>


<https://github.com/user-attachments/assets/c0ca76e5-afa2-42de-964c-a1fcec22eb48>


<https://github.com/user-attachments/assets/7fbd4884-f23a-4450-8fbf-f739a131ff13>

- Learning Results Videos
    - <https://drive.google.com/drive/folders/1ceDayoJZwuJBw2DGd_66w1x5YHSbzNAg?usp=sharing>
    - <https://drive.google.com/drive/folders/1cpIWT83J9RTWp3P_iqjMXLlou1uAx3C0?usp=sharing>
 

---


## Conclusion

```
✅ The values ​​learned using the vehicle license plate maintained a value of 0.8 to 0.9, showing high accuracy.

However, as it recognizes similar white bricks and lights, various license plates and a lot of data are needed.

However, the vehicle license plate image used for learning maintained a value of 0.8 to 0.9, showing high accuracy,

so training the model with more diverse license plate photos and angle data and applying

appropriate data processing resulted in improved values. You will get it.
```
