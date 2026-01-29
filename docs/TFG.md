---
title: "Macro-Nutrient Prediction of Food Images"
author: "Senén Marqués"
titlepage: true
date: 'January 2026'
geometry: "margin=0.8in" 
urlcolor: blue
linkcolor: blue
lang: "en"
mainfont: 'Montserrat'
monofont: 'JetBrainsMono Nerd Font'
listings-disable-line-numbers: true
listings-no-page-break: true
---

# Aim
The aim of this work is to explore and experiment with different methods of predicting macro nutrient information (grams of proteins, carbohydrates and fats) from images of food.

I'd like to start with a strong theoretical understanding of the problem. 
For that, I plan to:

- __Perform an EDA of food data:__ 
  - How is nutritional information accross different foods distributed?
  - Knowing this, what accuracy can we expect to obtain with a baseline model?
- __Test the accuracy of different pre-existing solutions:__ How good are pre-existing solutions, and can we compete with the SOTA?
- __Explore the different ways in which this problem is tackled:__ What architectures do SOTA solutions use?
- __Weight the different options and build prototypes__

# The Dataset
We are using the [MM-Food-100K Dataset](https://huggingface.co/datasets/Codatta/MM-Food-100K), which contains ~100K labeled images of home cooked food. Although we might change the dataset or use more than one throughout our work, this dataset will suffice for now.

# A Priori Knowledge
A priori, we try to tackle the problem with two approaches

## Using a Pipeline
We can build a pipeline using various pretrained models to:

1. Perform segmentation with classification of an image 
2. Predict volumes of each classified segment
3. Use the classifications and the volumes to calculate the micro nutrient profile and calories using a nutrition database

We will need to:

1. Test the pretrained models individually
2. Perhaps attempt transfer learning to suit our needs or look for more specialised models
3. Ensure each component is robust enough
4. Build the pipeline

We can use models like FastSAM for segmentation, YOLO26 for segmentation with classification and others for volume estimation.
However, the problem with this approach is that any error from a step can compound over to the next step in the pipeline, and if the individual components are not robust or accurate enough, we cannot expect good results.

## Training a Model on Labeled Images
Regardless of the model or architecture used, this is a much simpler and direct way of confronting the problem.
The benefit of this approach is that it is a single point of failure and it's viability depends entirely on how fine tuned our data and model are.
The strenghts of this approach can also be its weaknesses, as we need an adecuate dataset and model

1. Explore different datasets and perform an EDA
2. __Investigate:__ Can a model predict three values from an image or do we need to train three models to predict one value each?
3. Find a baseline / dummy model
3. __Explore SOTA models and architectures:__ How well do they fare against simpler/older architectures?

