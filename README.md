# tensorflow_lite_in_aws_lambda_function
using tensorflow's official tflite_runtime package, small env under 25mb




#### Making a Tensorflow model and deploying via a AWS-Lambda-Function can be done in the follow two steps:

1. Use a Python Notebook (or other method) to create a model using Tensorflow and Keras and which is then saved as a model.ftlite file.

2. Store the model in AWS-S3 and run the model using an AWS-Lambda-Function using a python env containing tflight_runtime. 
