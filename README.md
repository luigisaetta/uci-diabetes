# A model for predicting the duration of hospitalization in diabetic patients

February 2021

In this repository there is the code I have developed for a project with the goal of developing a model for the prediction of the hospitalizazation for patients with complicance due to diabetes.

The code is based on a modified version of the UCI diabetes dataset.

## Features:
* A model entirely based on Neural Networks: the model use a fully connected neural network, for regression
* Data transformations are 'inside the model', using TF 2.3 DenseFeatures and Feature Columns API
* The model predicts also the 'incertainty' on the predictions, using TF probability
* Explanation on how to best choose the threshold, to select a group of patients for a trial
* Bias analysis, using AEquitas toolkit



 
