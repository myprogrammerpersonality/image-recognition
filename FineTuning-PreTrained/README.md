# MultiClass Classification of images using Pretrained models from Torchvision

This folder uses pre-trained models of the TorchVision package to accelerate training on a new image dataset and use the pre-trained model as a feature extractor. 

It replaces the last layer of the model (classifier layer) to adjust it to our problem number of classes.

## input data requirements:
Images should be placed in seprate folder based on train/val/test. 

In each folder should be a file called label.csv that has two column:
1-image path 2-image label

image name doesn't matter!

MultiClass Classification uses CrossEntropyLoss.
