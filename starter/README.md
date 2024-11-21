# Inventory Monitoring at Distribution Centers project using amazon bin image dataset

 This is the final project for the Udacity's AWS Machine Learning Engineer Nanodegree, to show all the concepts learned during the program applied to resolve a real world problem. The problem to resolve is inventory monitoring (object counting) for bin images in distribuiton centers. In this scenario we are using CNN with transfer learning using Pytorch and AWS Sagemaker to perfom an image classification of the number of items present in an image.

## Project Set Up and Installation
This project requeres to have access to an AWS account.
The main file is the sagemaker.ipynb notebook who is spected to be executed in a jupyter notebook in aws sagemaker (in a stand alone instance or within sagemaker studio)

## Dataset

### Overview
To complete this project we will be using the Amazon Bin Image Dataset. The dataset contains 500,000 images of bins containing one or more objects. For each image there is
a metadata file in json format containing information about the image like the number of objects, it's dimension and the type of object. For this task, we will try to classify the
number of objects in each bin.

### Access
The dataset and the instructions to download it can be found in https://registry.opendata.aws/amazon-bin-imagery/

## Model Training
The model selected for this problem is the Pytorch implementation of EfficientNet, for being a better alternative to ResNet in terms of efficiency and accuracy with smaller datasets, since we are perfoming the training only over a smaller image subset from the Amazon dataset. The list can be found in the file_list.json file.
Over this model we will be perfoming the transfer learning technic, to detect the number of objects in a bin image in a range from 1 to 5 items (no zero or more than 5 items cases), since this is considered to be a "moderate" complecity task over the whole dataset being a "hight" complecity task.

## Machine Learning Pipeline
The basic flow for this problem can be found in the sagemaker.ipynb notebook
1. Download and perform EDA over the amazon dataset.
2. Split and upload the resulting datasets to s3
3. Perform training with hyperparameter tuning
4. Perform model training with best hyperparameters found
5. Model evaluation ( test inferences, visualize profiler reports and metrics)

## Standout Suggestions
For this project I've perfomed this suggestions:
- Multi instance training for a better training performance (2 instances)
- Hyperparameter tuning previous training, to find best parameters for a higher model accuracy. In this case the selected three are: learning rate, epochs and batch size, following the report for this problem in https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/blob/master/ProjectReport.pdf
- Model deployment for testing inferences over images
- Profiling and debugger in the traning process for better evaluation of the model and detect problems
