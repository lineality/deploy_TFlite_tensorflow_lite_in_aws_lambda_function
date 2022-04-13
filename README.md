# deploy_TFlite_tensorflow_lite_in_aws_lambda_function
- using tensorflow's official tflite_runtime package 
- small env under 25mb (easy to put in AWS and share with github)


## Create & Deploy Model in Two Project-Parts:

#### Making a Tensorflow model and deploying using an AWS-Lambda-Function can be done in the follow two steps:

Part 1. Use a Python Notebook (or other method) to create a model using Tensorflow and Keras and which is then saved as a model.ftlite file. The full Tensorflow and Keras and huge, can be difficult to install (so use colab...), and are for developers (not production deployment)

Part 2. Deploy and run in AWS using tflite: tflite is very small (and env less than 25mb) and works well for Edge and endpoint production deployment (using the same models produced using the full TF/Keras software suits). Store the model.tflite in AWS-S3 and run the model using an AWS-Lambda-Function using a python env containing tflight_runtime. 
- upload your model.tflite into AWS-S3
- create an AWS-Lambda-function, upload your python env
- (optionally) make an API-Endpoint with AWS API-Gateway for the AWS-Lambda-function
- (optionally) connect other AWS-lambda-functions or services to your AWS-Lambda-function

See items in repo:
- .ipynb notebook file.
- function.zip is the env to upload to AWS as is (this is a minimal env, not using all the packages used while making and testing the model)
- lambda_function.py is the specific code to customize if needed


### Instructions:

# Part One: Make your Model for TFlite (using full Tensorflow and Keras)

1. start google colab, a jupyter notebook, or other development environment.
#### Instructions for local jupyter notebook:
- in a terminal, in your project directory: 
```
$ python3 -m venv env; source env/bin/activate
```
- pip install tensorflow
- pip install matpotlib.pyplot
- pip install pandas
- pip install jupyter

2. Go through whatever process to make whatever model you are making in TF. The details vary based on what you are doing.

3. Important Note: Be careful to create your model.tflite file. Don't just stop with your TF model. 



```
#########################################
# Make Your TFlite Version of your Model
#########################################
# See: https://www.tensorflow.org/lite/convert/
 
# save the TF model as directory
TF_model_directory = 'saved_model/'
tf.saved_model.save(model, TF_model_directory)
 
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model( TF_model_directory ) # path to the SavedModel directory
tflite_model = converter.convert()
 
# Save the model.
with open('model.tflite', 'wb') as f:
 f.write(tflite_model)
 
# inspect
!ls
```
Though you can convert a model.h5 to model.tflite: 

https://github.com/lineality/TF_to_TFlite_convert_model.h5_to_model.tflite


4. Optional Step: Do a local test of your model.tflite file. (Always good to test before moving on.) The materials for such as test are included in this repo:
- pre-made zipped env for python 3.8 (or make your own)
- python script to run
- include your own tflite.model file


# Part Two: Deploy your model and AWS-Lambda-function-env in AWS

1. create a new AWS lambda function
2. upload your env to AWS lambda
3. use AWS api-gateway to make a public or private endpoint for the Lambda Function

Note: For computer vision your input may be a picture file in S3, vs. a number input for a regression model

```
Sample input:
{
  "s3_file_name": "model.tflite",
  "s3_bucket_name": "api-sample-bucket1",
  "user_input_for_X": 3.04
}
```

## Step 5: Yet another Python Environment -> Test your AWS deployment function code

To test out and or develope your deployment TFlite function details, you may want to create a 3rd python environment (unless you want to test directly in AWS, but that can be time consuming due to vague, erroneous, and absent error messages). 

To make a local python environment to test or deploy: 
```
$ pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Files for testing are provided here:

https://github.com/lineality/deploy_TFlite_tensorflow_lite_in_aws_lambda_function/tree/main/local_testing

re-Zip instructions are here, and here:

https://github.com/lineality/notes_on_zip 

https://github.com/lineality/linux_make_split_zip_archive_multiple_small_parts_for_AWS 



