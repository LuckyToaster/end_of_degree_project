---
title: "Macronutrient Mass Prediction from Food Images"
author: "Senén Marqués"
fontsize: 10pt
date: \today
bibliography: refs.bib
csl: ieee.csl
link-citations: true
linestretch: 1.5
mainfont: "Libertinus Serif"
sansfont: "Libertinus Serif"
monofont: "Libertinus Mono"
mathfont: "Libertinus Math"
header-includes:
  - \setkomafont{disposition}{\bfseries}
  - \usepackage{float}
  - \usepackage{graphicx}
geometry:
  - top=2cm
  - bottom=2cm
  - left=2cm
  - right=2cm
---

# Aim
The goal of this work is to systematically explore ways in which to predict the macronutrient mass (in grams) from food images. 
Macronutrient mass (grams of protein, fat and carbohydrate) were chosen as the three targets and they were used to calculate calories.
Predictions were made from 1) a single image and 2) no reference object for size in the image. These rules served to limit the scope of the project, 
to allow for the use of the same dataset, and to make a general solution that is suitable for real world applications. 

Achieving SOTA or close to SOTA quality presented a significant challenge. In order to accomplish this,
the project adopted an experimental and research driven approach, 
using the literature to justify decisions and hypotheses, and testing those hypotheses through experimentation,
thus adhering to the scientific method.

The source code for this project can be found at [https://github.com/LuckyToaster/end_of_degree_project](https://github.com/LuckyToaster/end_of_degree_project)

# Introduction

The biggest challenge when it comes to using computer vision models to predict nutritional information from food images is that it is difficult to predict mass or volume from food images,
using a reference object in images may help models better learn this feature @food_portion_size_estimation. @food_portion_size_estimation reported low error in predicting volume from food images,
although useful in demonstrating how to create a nutrition dataset (one containing reference objects like coins), did not disclose how they obtained such results and simply called it 
'a computer algorithm'. Since no dataset containing images with reference objects annotated with nutritional information was found from an early stage, the present work shifted to make a general purpose solution that would hopefully
predict low enough errors that would beat human estimations.

The researched on model architectures for this work was limited to an area of research involved in small and efficient architectures. @squeezenet studied how different filter dimensions affected network accuracy and 
efficiency among other things, they introduced _SqueezeNet_, a CNN with AlexNet level accuracy and 50x fewer parameters. They showcase three strategies to decrease the number of parameters in a network while preserving the most accuracy. 
The strategis showcased in @squeezenet are: First, replacing 3x3 filters with pointwise convolutions _"since a 1x1 filter has 9X fewer parameters than a 3x3 filter"_. Second, in order to shrink the parameter count further, decrease the number
of input channels or features that lead to 3x3 filters, as they highlight that the number of parameters in a layer are (number of input channels) * (number of filters) * (filter dimensions, eg: 3x3). Thirdly, downsampling late in the network as opposed 
to early, so that the final activation maps don't loose too much resolution and information is not lost, which according to @squeezenet is done mainly to preserve accuracy.

The parameter shrinking strategies showcased by @squeezenet led them to come up with the _Fire module_, of which SqueezeNet is mostly comprised of. A Fire module is made up of a _squeeze_ phase that uses 1x1 filters to shrink the depth of the output tensor depending on the number of 1x1 filters, 
and an _expand_ phase featuring 1x1 and 3x3 filters. So their fire module has three hyperparameters, the number of 1x1 filters in the squeeze, the number of 1x1 filters in the expand and the number of 3x3 filters in the expand phase. They note 
that in order to align with their second strategy, they keep the number of filters in the squeeze phase to be less than the sum of all the filters in the expand phase.

Later, @senetworks introduced the _Squeeze and Excitation_ (SE) block as a way to perform feature (channel) recalibration _"to selectively emphasise informative features and suppress less useful ones"_ @senetworks. It used the findings of @squeezenet to make an
improved module called the _"Squeeze and Excication block"_, which does feature recalibration by some sort of bypass or residual connection. Similar CNN architectures like @mobilenet, @mobilenetv3 and @efficientnet  were researched to gain a better knowledge of 
the evolution of Deep Convolutional Neural Networks and a deeper intuition as to what elements constitute an powerful yet efficient architecture.

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

Finally, upon revisiting the code months later, it was evident that having almost two hundred gigabytes of images locally was not a good idea,
so the data loading script was refactored and improved one final time so that images would be downloaded and resized in memory and then saved to disk. 
This cut the dataset size into a fraction of what it was before and improved the speed of the script somewhat as it made less I/O operations.

## A Quick EDA

After loading the data, it was manually checked for coherence. 15 samples of the dataset were chosen and their calorie and macronutrient annotations were checked so that the calory and macronutrient annotations matched.
Many rows were also inspected to see the type of images present and the quality of the other annotations. Then, calorie and macronutrient histograms were plotted to see which targets hold more weight and if the distribution of 
the data in the dataset is representative of real data, at least at a glance, which is useful when training a model. Below is a sample of the MM-Food-100K dataset, and the corresponding histograms.

\begin{figure}[H]
\centering
\includegraphics[width=1.0\textwidth]{dataset_sample.png}
\caption{Dataset Sample}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=1.0\textwidth]{food_distribution.png}
\caption{Calorie and Macronutrient Histograms}
\end{figure}


# Multi-ouput Regression
The computer vision task at hand requires multi output regression. 
A Neural Network can be adapted for regression by modifying the head of the network
and changing the criterion from Cross-Entropy loss to MSE (L2) or MAE (L1) Loss, 
_"Standard backbone networks are modified by replacing the final classification layer with a regression head"_ @from_pixels_to_cals.

The literature on CNNs is very classification biased, as it is centered around benchmarking on ImageNet classification.
After reading much literature on CNN architectures, the question of whether any ImageNet pretrained classification CNN would work well for our regression task emerged.
@do_imagenet_models_transfer_better analysed how 16 model architectures perform when fine tuned, trained from scratch or used as feature extractors
for 12 different datasets. They found that:

- Imagenet accuracy is a good indicator of transfer accuracy
- On some small, fine grained datasets, the benefit of fine-tuning vs training from scratch is minimal.
- The benefits of ImageNet pretraining fade for larger datasets.
- ImageNet pretraining accelerates convergence.

With this information, it was decided to use transfer learning when training the model, because even if the MMFood100K dataset is too large or too 
different from ImageNet, it will converge faster cutting down on training time.

## Model Architecture Evaluation
Before using transfer leaning to train a model to do multi-output regression, it was decided to make some empirical experiment to predict which
model architectore would perform better.

To find which pretrained model architecture would transfer better, or in other words, which would be the most suitable
model for our data and regression task, a _"model shootout"_ study was devised. 

A study was made with the [Optuna](https://optuna.org/) library that would pick different pretrained models using grid sampling, train the models with the same hyperparameters, 
for a fixed number of epochs using feature extraction by freezing the backbone and training only the head.

This was a cheap and fast way to make an educated guess to pick the model since _"when [different models are] used as fixed feature extractors ..., ImageNet top-1 accuracy was 
correlated with accuracy on transfer learning"_ @do_imagenet_models_transfer_better .

However, after seeing the terrible results of said study, and how the models did not learn at all, the _"model shootout"_ experiment was updated to run a study for each model to test, 
for a fixed number of trials, keeping all hyperparameters the same except the most critical, the learning rate. Since how well each architecture learns is mostly affected by 
the learning rate, it was set as the only hyperparameter to be optimized by optuna, then the best trial for each architecture would be selected and the model architecture with the best learning curve would be picked.

The experiment was ran to test the following models:

- EfficientNet_B3
- EfficientNet_V2_S
- MobileNet_V3_L
- Swin_V2_S

The variant (eg: B3, S, L) of each model was chosen as the biggest variant that could fit on a _16GB VRAM RTX 5060 ti_ GPU. 
After obtaining the results, the Swin vision transformer was chosen because its validation accuracy was slightly higher than the others and because it seemed like the most modern and promising architecture, besides that, 
there was barely any difference between the learning curves except perhaps a little less overfitting from the Swin model. The results are shown in the figure below.


\begin{figure}[H]
\centering
\includegraphics[width=1.0\textwidth]{model_shootout_grid.png}
\caption{Model shootout results}
\end{figure}

## Sequential fine tuning with Hyperparameter Optimization

After the Swin model won the shootout, a two step training stage was ran in an extensive hyperparameter optimization study to find 
the best settings for the model and minimize the loss. The the two step training, or sequential fine tuning, consisted of a _warm up_ step before fine tuning, where the head learns while the backbone is frozen.
The training loop in both studies featured Automated Mixed Precision Traninig, which was implemented after learning about it through @mixed_precision_training to attempt to train faster and with a lower memory foootprint, and was implemented following an official pytorch tutorial.

The following parameters and hyperparemeters were optimized by the optuna study:

- Feature extraction phase learning rate
- Feature extraction weight decay
- Feature extraction epochs
- Fine tuning phase learning rate
- Fine tuning weight decay
- Fine tuning epochs
- Loss used (MAE, MSE or Huber) 
- number of hidden units before the head (not a hyperparameter)
- hidden layer dropout (not a hyperparameter)

On the figure below is the learning curve of the best trial

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{fine_tuning_best_trial.png}
\caption{Fine Tuning Best trial}
\end{figure}

After the study, the model was trained using the best trial parameters and evaluated using a hidden validation split (a 3 way split was carried out throught the project in case the model was overfitted to the validation dataset).
The results, are illustrated below. 


| Nutrient | MAE (grams) | MAE (kcal) |
| :--- | :--- | :--- |
| Fat | 4.22 | 37.98 |
| Carbs | 9.68 | 38.72 |
| Protein | 4.8 | 19.2 |
| **Total** | - | **95.9** |


Overall, it was surprising to see s mean average error that low, which suggested that the model could be used as a real world calorie tracking system, albeit a not so precise one, in which a user could over or undereat around 300 calories a day, assuming logging three images 
daily for breakfast, lunch and dinner. However, these results can beat the nutrion estimates of the average person.

# LMM enriched with metadata

@lmms_and_acedata found that when feeding an image of food enriched with metadata 
like GPS coordinates, time of day and a list of ingredients to an LMM (Large Multimodal Model) at prompt time, 
while using prompting techniques like Chain of Thought, Scale Hint in the image, Few-shot and Expert Persona, 
the error in the predictions would shrink significantly. However, their reported MAE was too high to make such a solution usable.

Nonetheless, the LMM pipeline for estimating nutritional information in @lmms_and_acedata was replicated, 
in order to compare the results with the fine tuned vision transformer. It's inner workings are illustrated in the figure below. 

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{lmm.png}
\caption{LMM enriched with metadata}
\end{figure}

It was implemented through the use of the Langchain python module, and it consisted of the following components:

- An ingredient VQA (Visual Question Answering) LMM
- A coordinate fetching function
- A timestamp function
- A nutrition prediction LMM

The ingredient VQA LMM was simply an LMM that is given a food image and outputs a list of detected food items or ingredients. The list of ingredients, location coordinates and timestamp are fed to the nutrition prediction LMM
at prompt time to make the final prediction. Google's gemini-3.1-flash-lite LMM was used for both models in the pipeline. Since the food images in the dataset are not annotated with timestamps or information that
can be used to infer if the food was meant to be consumed as breakfast, lunch or dinner, the location and time information was not used to test the pipeline. 
Also, testing the pipeline on the whole hidden validation split would haved incurred thousands of API calls, 
which could have turned out too costly, so the evaluation of the pipeline was done with 100 samples of the hidden validation split. 
The results obtained were practically equivalent to those reported in @lmms_and_acedata, and can be seen below.


| Nutrient | MAE (grams) | MAE (kcal) |
| :--- | :--- | :--- |
| Fat | 12.47 | 121.23 |
| Carbs | 18.77 | 75.08 |
| Protein | 13.5 | 54 |
| **Total** | - | **250.31** |


# Future Improvements

While the results of the fine tuned Swin model were not too unsatisfactory, there is much room for improvement. 
First, both optuna studies could have been ran for much longer to obtain better results. The results of one promising fine tuning study with over 50 completed trials was accidentally lost, and the second time that
the study was ran, the range of the number of epochs was modified mid study, which caused optuna to perform suboptimally, yielding no better studies as it ran sampling the new search space. So it would have been ideal to have 
both hyperparemeter optimization experiments running properly and for much longer. In addition, better results could be achieved theoretically by removing outliers from the dataset, 
and performing data augmentations like in @autoaugment and @randaugment. Moreover, it is highly plausible that the studies for selecting a suitable architecture and fine tuning a model can be further improved.
And finally, more experiments could be carried out, perhaps by training more models on different targets, and investigating which combination yields the best results. Some future ideas could be:

- One model to predict each macronutrient mass
- One model to predict total mass and another to predict the distribution of the three macronutrients (although this does not address varying calorie densities of different foods)
- A pipeline like one described in @from_pixels_to_cals _"[A standard] Pipeline typically consists of food detection, classification, and portion estimation"_

These are important considearations that should be addressed in future work which could yield surprising results.  

#  References

