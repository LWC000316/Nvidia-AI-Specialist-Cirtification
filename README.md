![NVIDIA LOGO](https://github.com/user-attachments/assets/9cf87f01-ff75-4c6a-b4c8-2560ca2e4db7)

# Nvidia AI Specialist Certification
### <span style="color:violet">Title : Vehicle license plate recognition system using YOLOv5</span>
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
✅ Recent developments in autonomous driving technology have attracted more attention with
artificial intelligence-based recognition systems combining LiDAR and cameras, and these technologies
are playing an important role in a range of applications such as size plate recognition for vehicles
(automatic license plate recognition, ALPR).
```
---
## General description of the current project
```
✅ This project is to develop a vehicle license plate recognition system using YOLOv5.
It recognizes the vehicle's license plate as a partner, acknowledges it, and performs various functions
such as tracking stolen vehicles and automatic parking management.
```
---
## Proposed idea for enhancements to the project
```
✅ YOLOv5 processes the entire image at once to achieve accurate object detection while maintaining
high frame rates. This allows license plate recognition in real time even in environments where
vehicles are moving quickly.
```
---
## Value and significance of this project
```
✅ This project can be used in various fields such as traffic management, law enforcement,
and parking management through vehicle license plate recognition, and is an essential element
for autonomous vehicles to comply with traffic rules in real time and identify violative vehicles.
```
---
## Current limitations
```
✅ Limitations faced include poor recognition performance in various environmental conditions.
For example, when driving in bad weather or at night, it is difficult to accurately recognize license plates
using cameras alone, and vehicles moving at high speeds or partially obscured license plates
are highly likely to cause errors in existing systems.
```
---
## Literature review
```
✅ Because there are limitations such as dependency issues on large-scale datasets and poor performance
in various environments, improvements are needed by securing more datasets and using additional sensor
fusion technologies such as LiDAR.
```
---
## <span style="color:blue"> Image Acquisition Method </span>
- The parking lot black box video was filmed, and additional video
  was referenced from the Seoul road test driving video.
  

[DRIVE TEST](https://github.com/user-attachments/assets/9bfaefa1-c508-4fa7-a04f-94441b3b1514)


## <span style="color:blue">Learning Data Extraction and Learning Annotation </span>

- In order to learn with 640 x 640 resolution images in YOLOv5,

  the images were first created as 640 x 640 resolution images.


## <span style="color:blue"> Video Resolution Adjustment </span>


<https://online-video-cutter.com/ko/resize-video>

![비디오 리사이저](https://github.com/user-attachments/assets/ad1b5ca9-80a2-4b98-8716-a5ba8fd9276b)

- I used Darklabel to create edits with images based on frames, due to 640 x 640.
  

[DarkLabel2.4.zip](https://github.com/user-attachments/files/17794875/DarkLabel2.4.zip)

![darklabel 5](https://github.com/user-attachments/assets/1769e2b0-84ba-4854-beaa-2e4dd4cecf4c)

- First, add classes through darklabel.yml before annotation.
  

![code](https://github.com/user-attachments/assets/02742552-ec7a-46ca-a247-6e4b59594136)

- Add vehicle classes in the yaml file and add vehicle license plate as the class name.
  

![code 1](https://github.com/user-attachments/assets/6bbd6ffb-0b4b-434d-8fa1-1358e8020bb3)

- When annotating, put the vehicle classes in classes_set so that you can see the classes set

  in the DarkLabel GUI, and set the name to be viewed in the GUI to vehicle license plate.
  

![darklabel 2](https://github.com/user-attachments/assets/fb3cb492-bf57-4077-b8c3-864958c4b68e)

- You can see that classes called vehicle classes have been added to the DarkLabel program,

  and a vehicle license plate has been added below.
  

![darklabel 4](https://github.com/user-attachments/assets/c467f5de-d811-47d4-902a-5e37352298ac)

- In the DarkLabel program, you can convert video into images frame by frame.

  First, select a 640 x 640 resolution video through Open Video. Afterwards,

  it is converted to an image through as images, and the labeled value is saved through GT save.

<img src="https://github.com/user-attachments/assets/04b3735c-1215-4ccb-abef-f59a1bb37117" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/ac618c57-f31d-4164-99e1-741cfd537b6a" width="50%" height="50%">
    
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

![code](https://github.com/user-attachments/assets/9bd9a091-8062-472c-b0e5-5adf6d420aab)

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

<img src="https://github.com/user-attachments/assets/9cb71cfd-7045-4850-81b1-5e4251046d5b" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/157c0bc7-a822-468c-97c5-de6f4bc143e8" width="50%" height="50%">

- P_Curve / R_Curve

<img src="https://github.com/user-attachments/assets/684c724f-9c31-4f66-9ff8-9c8480c0e879" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/2c0516dd-aaaa-4fa6-a09e-f9d152d9613f" width="50%" height="50%">

- confusion_matrix

<img src="https://github.com/user-attachments/assets/93f2e9eb-11d2-48b0-93e3-0fbfb20951cf" width="50%" height="50%">

- labels / labels_correlogram

<img src="https://github.com/user-attachments/assets/206fca60-0bd9-4437-a5c6-c3e8054ef091" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/808d3efa-fbda-4fc6-804c-b564912e8d31" width="50%" height="50%">


- results

![results](https://github.com/user-attachments/assets/2bc6a3e6-21d1-477b-af9a-68a561590eb0)


- val_batch1_pred / val_batch2_pred

<img src="https://github.com/user-attachments/assets/8b1b8408-3227-4902-90ab-ab212bfb9274" width="50%" height="50%"><img src="https://github.com/user-attachments/assets/27cf293f-9ebf-40c8-ace4-3812761e4559" width="50%" height="50%">

- learning file

    - [exp.zip](https://github.com/user-attachments/files/17817104/exp.zip)

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
