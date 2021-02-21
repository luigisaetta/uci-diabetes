# A model for predicting the duration of hospitalization in diabetic patients

February 2021

In this repository there is the code I have developed with the goal of developing a model for the prediction of the hospitalization for patients with complicance due to diabetes.

The code is based on a modified version of the UCI diabetes dataset (https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).

## Features:
* A model entirely based on Neural Networks: the model use a fully connected neural network, for regression
* Data transformations are 'inside the model', using TF 2.3 DenseFeatures and Feature Columns API
* The model predicts also the 'incertainty' on the predictions, using TF probability
* Explanation on how to best choose the threshold, to select a group of patients for a trial
* Bias analysis, using AEquitas toolkit

## References
@article{2018aequitas,
     title={Aequitas: A Bias and Fairness Audit Toolkit},
     author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid}, journal={arXiv preprint arXiv:1811.05577}, year={2018}}


 
