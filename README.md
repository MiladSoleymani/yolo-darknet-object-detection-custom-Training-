# Train YOLOv4 using darknet

![download](https://user-images.githubusercontent.com/78655282/128880791-374b1bee-94cb-4ff0-98a4-1dc150a2ed0d.png)


YOLO stands for You Only Look Once. YOLO is a state-of-the-art, real-time object detection system. It was developed by Joseph Redmon. It is a real-time object recognition system that can recognize multiple objects in a single frame.

## How YOLO works

YOLO uses a totally different approach than other previous detection systems. It applies a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
The basic idea of YOLO is exhibited in the figure below. YOLO divides the input image into an S × S grid and each grid cell is responsible for predicting the object centered in that grid cell.


![0_Okuwq93g3v13CShN](https://user-images.githubusercontent.com/78655282/128882771-c7c9b824-f8d9-4a82-96c3-9e42d0584a50.jpg)


Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.


![0_IhbtJNWpPG1PgTRk](https://user-images.githubusercontent.com/78655282/128882941-3cd7a591-96a1-463e-95c9-c5aad67cb251.png)


## About YOLOv4 

YOLOv4 is an object detection algorithm that is an evolution of the YOLOv3 model. The YOLOv4 method was created by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. It is twice as fast as EfficientDet with comparable performance. In addition, AP (Average Precision) and FPS (Frames Per Second) in YOLOv4 have increased by 10% and 12% respectively compared to YOLOv3. YOLOv4’s architecture is composed of CSPDarknet53 as a backbone, spatial pyramid pooling additional module, PANet path-aggregation neck, and YOLOv3 head.
YOLOv4 uses many new features and combines some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a real-time speed of ~65 FPS on Tesla V100. Following are the new features used by YOLOv4:

- Weighted-Residual-Connections (WRC)
- Cross-Stage-Partial-connections (CSP)
- Cross mini-Batch Normalization (CmBN)
- Self-adversarial-training (SAT)
- Mish activation
- Mosaic data augmentation
- DropBlock regularization
- Complete Intersection over Union loss (CIoU loss)

![1_m2N0pah1h1fYKlkAYGlsLQ](https://user-images.githubusercontent.com/78655282/128883815-3f1618fb-9636-4a85-b0e5-1876f5f5a075.png)

## For more information, you can use the following link.

- [You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)


## How to train YOLOv4?

I have prepared a pipeline to train your custom datasets using YOLOv4 and Tiny-YOLOv4, and I have explained everything from preparing datasets to getting results and inference. You can find the training pipeline in train_YOLOv4.ipynb and train_tiny_YOLOv4.ipynb. In these, notebooks I have used YOLOv4 and Tiny-YOLOv4 to detect screws on the conveyor belt and count them. Can be seen the results in below :


## YOLOv4 results

![chart](https://user-images.githubusercontent.com/78655282/195811592-a37f5b99-eaff-454d-a643-eab2c7b217b2.png)


## Tiny-YOLOv4

![download (2)](https://user-images.githubusercontent.com/78655282/195811674-8b7ccb6b-4dbf-481b-be65-2497536df26c.png)


## Final infrence from YOLOv4

![yolov4 out](https://user-images.githubusercontent.com/78655282/129019805-7d236b7c-b8c6-46b7-9b61-50abc128718f.png)

