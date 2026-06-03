---
title: "Macronutrient Mass Prediction from Food Images"
author: "Senén Marqués"
fontsize: 10pt
date: \today
bibliography: refs.bib
csl: ieee.csl
link-citations: true
---

# Abstract

# Aim
The goal of this work is to systematically explore ways in which to predict the macronutrient mass (in grams) from food images.  Predictions 
were made from 1) a single image and 2) no reference object for size in the image. These rules served to limit the scope of the project, 
to allow for the use of the same dataset, and to make a general solution that is suitable for real world applications

Achieving SOTA quality presented a significant challenge. In order to accomplish this,
the project adopted an experimental and research driven approach, 
using the literature to justify decisions and hypotheses, and testing those hypotheses through experimentation,
thus adhering to the scientific method

# The Data
The dataset used was the **[MM-Food-100K Dataset](https://huggingface.co/datasets/Codatta/MM-Food-100K)**, which contains ~100K labeled images of restaurant and home cooked food.
It was chosen because the images are labelled with nutrionional information needed for this work, for of its large size, 
and because of its diversity: featuring different dish sizes, cuisines, lighting, camera-quality ... etc.

# 1st Experiment: Multi-ouput Regression using Vision Transformer

In order to determine which CNN architecture was best, a **model shootout** study was carried out comparing the following pretrained CNNs (and a vision transfoermer)
- EfficientNet_B3 
- EfficientNet_V2_S
- MobileNet_V3_L
- Swin_V2_S

The variant of each model was chosen as the biggest variant that could fit on a **16GB VRAM RTX 3060 ti** GPU. 
The study was performed by training all models with the same hyperparameters

The hyperparameters:
- **Batch size:** 32 (except for MobileNetV3_L which could accept a batch size of 256
- 

# 2nd Experiment:


# Hardware Limitations
In the last decade, the computer vision landscape has been filled with CNN's, however, in recent years, Vision Transformers have become SOTA. 
Different CNNs  

While we would like to eventually experiment with Vision Transformers, 
we are aware that they are computationally expensive and require larger datasets than what we own. 
Seeing it that we are working with a __4GB VRAM limitation__, for now we have directed our attention towards the SOTA in CNNs


# A-Priori Knowledge
A priori, we can tackle the problem with two approaches.

The first is to build a pipeline using various pretrained models to: 1) Perform segmentation with classification. 2) Predict volumes for each classified segment. 3) Use the the classifications and volumes to trivially obtain the macronutrient mass. 
We can use models like **FastSAM** for segmentation, **YOLO26** for segmentation with classification, and others for volume estimation.
However, the problem with this approach is that any error from a step can compound over to the next step in the pipeline, and if the individual components are not robust or accurate enough, we cannot expect good results.

The second is to simply train a model with our dataset of labeled images and predict macronutrient mass directly. Regardless of the data and CNN architecture used, this is a much simpler and direct way of confronting the problem and the results depend mainly on the dataset and CNN Architecture used.

Naturally, we opted for the second approach as a starting point.

# Methodology 
## Loading the Data
The first step was to download the _MM-Food-100K Dataset_, to do this, we made a python script to download the dataset csv file and images concurrently and idempotently, meaning that upon each execution, 
the script will attempt to download any missing images, as it will take several hours to download the whole **166 GB** of images and it is unrealistic to expect the whole script to finish execution uninterrupted in one go.

To ensure maximum speed, our data loading script features concurrency for networking tasks and parallelism for processing and disk access.

After having a local copy of the dataset, we made a custom pytorch Dataset class to allow obtaining the image and targets when iterating over it. This is the Dataset that will be passed to a DataLoader and then to the model.

When testing our custom Dataset class, we noticed warnings due to corrupted images and we modified our script to scan for corrupted files and remove them from the dataset. 
The script was able to detect some corrupted images but not all, which was frustrating.
So, although we were able to use our Dataset for training, we would like improve our data loading script to remove any images that will raise warnings from torchvision's internal JPEG handling logic.

Later during training, we realized there was a CPU bottleneck caused by the heavy image transform computations from the DataLoader loading batches into the model. 
We again modified our data loading script to do the heaviest transform, image Resizing, locally. Thus avoiding the CPU bottleneck, and allowing batches to be fed into the model at twice the speed.

We are very eccentric about our code, not only do we want it to be simple and pythonic, we also want it to be performant and reliable. 
We certainly don't want to see any warning or errors when loading images as we could very well be __feeding bad data to our model__. 
On a more vapid tone, we don't want our terminal to be polluted with warnings every time we run a script or train a model as it looks shabby.
For this very reason, we wrote some tests for our data loading script, and will continue revisiting it throughout our work to improve it.

## Choosing a CNN Architecture
We have a computer vision task which requires multi output regression, sadly, we noticed that the literature on CNNs is very classification biased. We are aware that we can adapt any CNN for regression by modifying the head of the network and changing the criterion from Cross-Entropy loss to MSE (L2) or MAE (L1) loss.
However, almost all the CNNs we reviewed in the literature were benchmarked on ImageNet classification and we don't know which is best suited for multi-output regression. 
We need to investigate more @do_imgnet_models_transfer_better, but as a starting place, we decided to 1) train from scratch and 2) fine tune an existing model.

The most notable one is **EfficientNet** @efficientnet, which outperforms many other CNNs in accuracy and computation efficiency by degrees of magnitude. 
Notably, its microarchitecture features modules made from an MBConv (mobile inverted bottleneck) @mobilenetv2 followed by an SE Block (Squeeze and Excitation Block) @senetworks and the MBConv features skip connections, all of which are innovations we have encountered in the literature.
Its macroarchitecture repeats these modules with different kernel sizes and number of channels.

EfficientNet's baseline model (b0) was found using _multi objective neural architecture search_ that optimises accuracy and FLOPS @efficientnet, consequent variants (b1 through b7) were obtained by applying a custom compound scaling method that finds the best constant values for depth, width and resolution and then scales up by a factor of $\phi$, 
efficiently scaling the baseline. We would have liked to train a larger variant of this network but we found ourselves limited by hardware once again.

## Training
We trained EfficientNet_B0 from scratch, after monkey-patching it to suit our multi-output regression needs. We trained it for a period of 50 epochs and obtained an L1 (MAE) loss of 0.15, which we multiplied by each output's standard deviation to de-standardize it, and found that our 0.15 loss meant an average error of 2.25 grams of fat, 4.35 grams of carbs and 2.55 grams of protein, which gaves us a mean error of ~48 Kcal.

We are currently training it for 100 epochs, and will soon obtain its validation error and plot the losses.
We are then interested in experimenting with the following:

- Training other models (MobileNetV3)
- Attempting fine tuning
- performing data agumentation
- Replicating the training configurations used in the EfficientNet paper @efficientnet


