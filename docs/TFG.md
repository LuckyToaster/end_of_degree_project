---
title: "Macronutrient Mass Prediction from Food Images"
author: "Senén Marqués"
fontsize: 10pt
date: \today
bibliography: refs.bib
csl: ieee.csl
link-citations: true
mainfont: "TeX Gyre Pagella"
sansfont: "TeX Gyre Heros"
geometry:
  - top=2cm
  - bottom=2cm
  - left=2cm
  - right=2cm
---

# Abstract

# Aim
The goal of this work is to systematically explore ways in which to predict the macronutrient mass (in grams) from food images. 
Macronutrient mass (grams of protein, fat and carbohydrate) were chosen as the three targets and they were used to calculate calories.
Predictions were made from 1) a single image and 2) no reference object for size in the image. These rules served to limit the scope of the project, 
to allow for the use of the same dataset, and to make a general solution that is suitable for real world applications. 

Achieving SOTA or close to SOTA quality presented a significant challenge. In order to accomplish this,
the project adopted an experimental and research driven approach, 
using the literature to justify decisions and hypotheses, and testing those hypotheses through experimentation,
thus adhering to the scientific method

# The Data

## Choosing a dataset
The dataset used was the **[MM-Food-100K Dataset](https://huggingface.co/datasets/Codatta/MM-Food-100K)** @mmfood100k, which contains ~100K labeled images of restaurant and home cooked food.
It was chosen because the images are labelled with nutrionional information needed for this work, for of its large size, 
and because of its diversity: featuring different dish sizes, cuisines, lighting, camera-quality ... etc.

Other researched datasets include ACEDATA @lmms_and_acedata, FooDD @foodd, Pittsburgh Fast Food Image Dataset @pittsburgh and Nutrition5K @nutrition5k. 
All of which, provided interesting insights into what makes a good dataset for the task in hand.

## Loading the data
The first step was to download the _MM-Food-100K Dataset_, to do this, 
a python script was made to download the dataset csv file and images concurrently and idempotently, 
meaning that upon each execution, the script would attempt to download any missing images, 
as it took many hours to download the whole 166 GB of images and it was unrealistic to expect the whole script to finish execution uninterrupted in one go.
To ensure maximum speed, the data loading script featured concurrency for networking tasks and parallelism for processing and disk access.

After having a local copy of the dataset, a custom pytorch Dataset class was made to allow obtaining the image and targets when iterating over it. 
That Dataset class would be passed to a DataLoader and then fed to the model.

Later during training, a bottleneck was found, it was caused by the heavy image transform computations from the DataLoader loading batches into the model.
The data loading script was modified to offload the image resizing computation to that data loading stage. 
After that improvement, batches were fed to the model during training at approximately twice the speed.

Finally, upon revisiting the code months later, it was evident that having alsmot two hundred gigabytes of images locally was not a good idea,
so the data loading script was refactored and improved one final time so that images would be downloaded and resized in memory and then saved to disk. 
This cut the dataset size into a fraction of what it was before and improved the speed of the script somewhat as it made less I/O operations.

# Multi-ouput Regression
The computer vision task at hand requires multi output regression. 
A Neural Network can be adapted for regression by modifying the head of the network
and changing the criterion from Cross-Entropy loss to MSE (L2) or MAE (L1) Loss.

The literature on CNNs is very classification biased, as it is centered around benchmarking on ImageNet classification.
After reading much literature on CNN architectures, the question of whether any ImageNet pretrained classification CNN would work well for our regression task emerged.
@do_imagenet_models_transfer_better analysed how 16 model architectures perform when fine tuned, trained from scratch or used as feature extractors
for 12 different datasets. They found that:

- Imagenet accuracy is a good indicator of transfer accuracy
- On some small, fine grained datasets, the benefit of fine-tuning vs training from scratch is minimal.
- The benefits of ImageNet pretraining fade for larger datasets.
- ImageNet pretraining accelerates convergence.

With this information, it was decided to use transfer learning when training the model, because even if the MMFood100K dataset is too large or too 
different from ImageNet, it will converge faster saving a of training time.

Sadly, @do_imagenet_models_transfer_better did not consider regression or predictions of continuous numerical values, 
and not much information was found on this topic besides some casual internet articles.


## Model Architecture Evaluation
Before using transfer leaning to train a model to do multi-output regression, it was decided to make some empirical experiment to predict which
model architectore would perform better.

To find which pretrained model architecture would transfer better with out dataset, or in other words, which would be the most suitable
model for our data, a _"model shootout"_ study was devised. 

A study was made with the [Optuna](https://optuna.org/) library that would pick different pretrained models using grid sampling, train the models with the same hyperparameters, 
for a fixed number of epochs using feature extraction (frozen backbone, training only the head).

This was a cheap and fast way to make an educated guess to pick the model since _"when [different models are] used as fixed feature extractors ..., ImageNet top-1 accuracy was 
correlated with accuracy on transfer learning"_ @do_imagenet_models_transfer_better .

After runnign the _"model shootout"_ optuna study for four times, it was determined that out of the following models:

- EfficientNet_B3
- EfficientNet_V2_S
- MobileNet_V3_L
- Swin_V2_S

The Swin came on top being the fastest learner, which makes sense since all the others are CNNs and Swin is a more modern Vision Transformer.

The variant of each model was chosen as the biggest variant that could fit on a _16GB VRAM RTX 3060 ti_ GPU. 

## Sequential fine tuning with Hyperparameter Optimization

After the Swin model won the shootout, a two step training stage was ran in an extensive hyperparameter optimization study to find 
the best settings for the model and minimize the loss.

The sequential fine tuning consisted of a _warm up_ step before fine tuning, where head learns while the backbone is frozen.

The sequential fine tuning was ran in an extensive hyperparameter optimization study using the Optuna library, that optimized the following hyperparameters:

- Feature extraction phase learning rate
- Feature extraction weight decay
- Feature extraction epochs
- Fine tuning phase learning rate
- Fine tuning weight decay
- Fine tuning epochs
- Loss used (MAE, MSE or Huber) 

And the following hidden flat layer that was inserted before the head

- number of hidden units before the head (not a hyperparameter)
- hidden layer dropout 


# LMM enriched with metadata

While researching the task at hand, @lmms_and_acedata found that when feeding an image of food enriched with metadata 
like GPS coordinates, time of day and a list of ingredients to an LMM (Large Multimodal Model) at prompt time, 
while using prompting techniques like Chain of Thought, Scale Hint in the image, Few-shot and Expert Persona, 
the error in the predictions would shrink significantly. However, @lmms_and_acedata reported MAE too big to make 
an accurate food logging system out of it.

A system was devised, that consisted of:
- A pretrained food classifier
- A coordinate fetching function
- A timestamp function

![LMM enriched with metadata](lmm.png)

# Results

**[I currently have no results to showcase]**
**[I will update the document with more sections, references and result visualizations]**



# References

