# Finetuning Object Detection Model from TorchVision
There is two sub-task here: 
1. object detection and lebeling
2. object detection, labeling and segmentation.

for `1` you only need to change Dataset Object in a way that output the object location and label in an image, for `2` you also need to have a seperate folder with mask image.

Reference: [Pytorch Transfer Learning Object Detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
