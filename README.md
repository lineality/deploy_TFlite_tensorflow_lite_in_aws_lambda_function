# deploy_TFlite_tensorflow_lite_in_aws_lambda_function_plus_CV_computer_vision
- using tensorflow's official tflite_runtime package 
- small env under 25mb (easy to put in AWS and share with github)


## Create & Deploy Model in Two Project-Parts:

#### Summary: Here is a thrifty, cheap, and relatively quick way to deploy AI/ML models in AWS. You can use google-colab to make your model, and you can deploy on AWS using cheap or free-tier resources: AWS-S3, AWS-Lambda-Functions, AWS-api-gateway endpoint. No docker needed, no cloud9, no sagemaker, no EC2, etc. TFlite-runtime is a very tiny way to deploy full models developed with Tensorflow and Keras tools on "Edge" (resource thin) or in AWS-lambda-functions (as described here). Overall there are two parts to the process. One: train your model and save it for TFlite. Two: Deploy the TFlite model in AWS. 


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

5: Yet another Python Environment -> Test your AWS deployment function code

To test out and or develope your deployment TFlite function details, you may want to create a 3rd python environment (unless you want to test directly in AWS, but that can be time consuming due to vague, erroneous, and absent error messages). 


# To make a local python environment to test or deploy: 

Files for testing are provided here:

https://github.com/lineality/deploy_TFlite_tensorflow_lite_in_aws_lambda_function/tree/main/local_testing

re-Zip instructions are here, and here:

https://github.com/lineality/notes_on_zip 

https://github.com/lineality/linux_make_split_zip_archive_multiple_small_parts_for_AWS 


# CV Computer Vision:
For a computer vision project, the process is very similar with a few notable differences.
1. You need to add Pillow/PIL (Python Image Library) to your env, which is only adds about 7mb to the compressed size. (Which is lucky, because you cannot go over the arbitrary AWS size limit or this method is impossible.) 
2. You need to account for the 'input' being a picture in S3, so you have two access-S3 file events.
3. The output can take various forms of your choice. If the model is a binary classification between two items, you can:
1. Simply print the output (a very odd form of a list with a double-space separator)
2. Translate the probability numbers into the names of those classes
3. You can give a single prediction: 1 vs 0 chance of one class or the other.
4. Or you can return a probability number of one or
5. both classes. 
(Many more design choices here, compared with a continuous-number prediction.)

## Be careful of library and package version incompatibility. 
Because AWS error messages are...garbage...it can take hours or days or longer to find out what is breaking a process. I strongly recommend going step by agonizing-step and testing in AWS with every step. Every time you add a python package, start with a minimal lambda-function and import that package. This way you can catch when something breaks. E.g. TFlite-runtime v28 breaks AWS (for whatever reason), but v27 is just fine. 

## Don't Re-use Disposable Parts:
In AWS-land, most things are brittle and need to be remade from scratch (even S3 folder paths...amazingly). Don't re-use a lambda function after re-building it several hundred times getting it to work: make a fresh one. Don't re-use a python-env: make a fresh one. Problems very often come from parts being broken and needing to start from a fresh part. 

## Save old versions of everything, and clearly mark what builds work.
- use whatever versioning system you want, but save save save, so when you need to go back then you can do so.
- "Coding is about communicating": Document everything you do clearly both for other people and for 'future you.' 

# Env Creation Instructions:

Working env files are included in this repo, but eventually you will want to make your own. After doing so a few times, the process becomes rather quick. 

instruction code to create python env (for uploading to AWS):
only tflite_runtime is needed, numpy is included with tflite

```
$ python3 -m venv env; source env/bin/activate

$ pip3 install --upgrade pip
```

#### **version 2.8 may cause AWS errors, try TFlite-runtime v2.7**
#### use: pip install tflite-runtime==2.7.0
```
$ pip3 install pillow

$ pip3 freeze > requirements.txt
```

####  Drill down to -> env/lib/python3.8/sitepackages
#### this makes the main zip file
```
$ zip -r9 ../../../../function.zip .
```

#### Make a lambda_function.py file, later update this with real code
```
$ touch lambda_function.py
```

#### In project-root folder: add .py file to your zip file
```
$ zip -g ./function.zip -r lambda_function.py
```

#### To update the .py file (depending on OS) edit file in the zipped archive or re-add a new .py to replace the old by repeating the same step from above
```
$ zip -g ./function.zip -r lambda_function.py
```



