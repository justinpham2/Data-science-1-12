![](UTA-DataScience-Logo.png)

# Spaceship Titanic

* **One Sentence Summary** Ex: This repository holds an attempt to apply Keras to data from
"Spaceship Titanic" Kaggle challenge https://www.kaggle.com/competitions/spaceship-titanic. 

## Overview

* This section could contain a short paragraph which include the following:
  * **Definition of the tasks / challenge**  Task: The task, as defined by the Kaggle challenge is to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
  * **Your approach** Approach: The approach in this repository formulates the problem as logistic regression task, using deep recurrent neural networks as the model. 
  * **Summary of the performance achieved** Ex: Training did not work. 

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
![image](https://user-images.githubusercontent.com/98443119/207671719-e1d2a7b5-419f-4bdf-9212-72922485266f.png)


  4677 for testing, 8694 for training, none for validation

#### Preprocessing / Clean up

* To clean up:
* Checked for missing values
* Checked for duplicated values
* Concatenated the data

### Problem Formulation

  * The input is transported the output is True/False
  * Logistic regression because it is trying to determine 'Transported'
  * Loss: binary_crossentropy, Optimizer: Rmsprop, Metrics: accuracy

### Training
I used keras deep learning
Hardware failed
Used two layers
Used relu for activation
### Performance Comparison

N/A

### Conclusions

* Try other methods before deep learning.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

You could apply this method/package to medical datasets. 

### Overview of files in repository

Spaceship-titanic.ipynb

### Software Setup

numpy as np 
pandas as pd 
matplotlib.pyplot as plt
seaborn as sns
sklearn.linear_model import LogisticRegression
sklearn.metrics import roc_auc_score,accuracy_score
sklearn.neighbors import LocalOutlierFactor
sklearn.model_selection import train_test_split

### Data

Data can be found in kaggle link. 

### Training

Use kera's and try applying different layers. 

#### Performance Evaluation

*N/A


## Citations
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/







