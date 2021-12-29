# tensorflow_lite_in_aws_lambda_function
using tensorflow's official tflite_runtime package, small env under 25mb


# Create & Deploy Model in Two Parts:

#### Making a Tensorflow model and deploying using a AWS-Lambda-Function can be done in the follow two steps:

Part 1. Use a Python Notebook (or other method) to create a model using Tensorflow and Keras and which is then saved as a model.ftlite file.

See .ipynb notebook file.

Part 2. Store the model in AWS-S3 and run the model using an AWS-Lambda-Function using a python env containing tflight_runtime. 
- upload your model.tflite into AWS-S3
- create an AWS-Lambda-function, upload your python env
- (optionally) make an API-Endpoint with AWS API-Gateway for the AWS-Lambda-function
- (optionally) connect other AWS-lambda-functions or services to your AWS-Lambda-function

see:
- function.zip is the env to upload to AWS as is
- lambda_function.py is the specific code to customize if needed


# Instructions:

## Part One: Make your Model
1. start google colab, a jupyter notebook, or other development environment.
#### Instructions for local jupyter notebook:
- in a terminal, in your project directory: 
```
$ python3 -m venv env; source env/bin/activate
```
- pip install tensorflow
- pip install matpotlib.pyplot
- pip install pandas
- pip install jupyter notebook
- pip install black[jupyter]


Note:
maybe add tflite_runtime tester cell in the notebook...


Note: the format of the input (need numpy?)

Note: 




