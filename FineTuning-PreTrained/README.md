# MultiClass Classification of images using Pretrained models from Torchvision

This folder uses pre-trained models of the TorchVision package to accelerate training on a new image dataset and use the pre-trained model as a feature extractor. 

It replaces the last layer of the model (classifier layer) to adjust it to our problem number of classes.

## input data requirements:
Your image should be placed in separate folders based on the train/val/test set. In each folder, you need to have a `label.csv` file that has two-column, the first, Path, includes the path to images, and the second column contains the labels. 
