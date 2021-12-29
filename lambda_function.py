# code for AWS-Lambda-function to deploy model.tflite in AWS

"""
https://github.com/lineality/tensorflow_lite_in_aws_lambda_function
"""

"""
To install the tensorflow lite package (follow official tensorfow docs)
use:
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime


# instruction code to create python env (for uploading to AWS):
# only tflite_runtime is needed, numpy is included with tflite

$ python3 -m venv env; source env/bin/activate

$ pip install --upgrade pip

$ pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

$ pip freeze > requirements.txt


# drill down to -> env/lib/python3.8/sitepackages
# this makes the main zip file
$ zip -r9 ../../../../function.zip .

# make a lambda_function.py file
# later update this with real code
$ touch lambda_function.py

# In project-root folder:
# add .py file to your zip file
$ zip -g ./function.zip -r lambda_function.py

# to update the .py file (depending on OS)
# edit file in the ziped archive
# or re-add a new .py to replace the old by repeating the same step from above
$ zip -g ./function.zip -r lambda_function.py
"""



"""
Workflow:

1. get user_input
2. S3: Connect to S3 (Make resource and client)
3. download zip file from S3
4. extract zip and put in /tmp/
5. load model
6. make prediction
7. clear /tmp/
8. export result
"""

"""
Sample Ouput:
{
  "statusCode": 200,
  "about": "output_of_Tensorflow_ML_model",
  "body": -0.07657486200332642
}
"""

"""
Sample input: 
{
  "s3_file_name": "model.tflite",
  "s3_bucket_name": "api-sample-bucket1",
  "user_input_for_X": 3.04
}
"""

# import librarires
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter 

import pathlib  # not needed?
import numpy as np 

import boto3  # for AWS
import glob  # for directory file search
#from zipfile import ZipFile  # Optional

def get_file_from_S3_to_lambda_tmp(s3_resource, s3_bucket_name, s3_file_name, lambda_tmp_file_name):



    # s3.meta.client.download_file('mybucket', 'hello.txt', '/tmp/hello.txt')
    s3_resource.meta.client.download_file( s3_bucket_name, s3_file_name, lambda_tmp_file_name )
    
    return print("Model saved.")

# helper function to clear remaining .csv files from /tmp/ directory
def clear_tmp_directory():

    """
    requires:
        import os (to remove file)
        import glob (to get file list)
    """

    # use glob to get a list of remaining .csv files
    remaining_files_list = glob.glob("/tmp/*.csv")
      
    # File location
    location = "/tmp/"

    # iterate through list of remaining .csv files
    for this_file in remaining_files_list: 
        # Remove this_file
        os.remove(this_file)

    # AGAIN use glob to get a list of remaining .csv files
    remaining_files_list = glob.glob("/tmp/*.csv")

    return print("""/tmp/ cleared. Check that directory is empty. remaining_files_list = """, remaining_files_list )

# helper function 
def print_aws_tmp_files():

    """
    requires:
        import os (to remove file)
        import glob (to get file list)
    """

    # use glob to get a list of remaining .csv files
    aws_tmp_files_list = glob.glob("/tmp/*")
      
    return print( "/tmp/ files_list = ", aws_tmp_files_list )

def lambda_handler(event, context):

    #################
    # Get User Input
    #################

    # get s3_file_name and path in s3
    # Test for input:
    try:
        s3_file_name = event["s3_file_name"]

    except Exception as e:
 
        output = f"""Error: No input for s3_file_name 
        Error Message = '{str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }      

    # get s3_bucket_name in s3
    # Test for input:
    try:
        s3_bucket_name = event["s3_bucket_name"]

    except Exception as e:
 
        output = f"""Error: No input for s3_bucket_name 
        Error Message = '{str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }    


    # get user_input_for_X in s3
    # Test for input:
    try:
        user_input_for_X = event["user_input_for_X"]

    except Exception as e:
 
        output = f"""Error: No input for user_input_for_X 
        Error Message = '{str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }    


    ####################################
    # S3: Connect to S3 (Make resource)
    ####################################

    try:
        # make s3_resource
        s3_resource = boto3.resource("s3")

        # make S3 bucket-resource
        s3_bucket = s3_resource.Bucket(s3_bucket_name)


    except Exception as e:
 
        output = f"""Error: Could not connect to AWS S3.
        Error Message = '{str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }


    ##################################
    # load file from S3 int /tmp/
    ##################################

    # AWS Files Name
    lambda_tmp_file_name = "/tmp/" + s3_file_name

    try:
        ###################
        # Get file from S3
        ###################
        get_file_from_S3_to_lambda_tmp(s3_resource, s3_bucket_name, s3_file_name, lambda_tmp_file_name)


    except Exception as e:

        output = f"""Error: Could not get data .csv file from S3
        Error Message = {str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }


    # for terminal: see what files exist in /tmp/
    print_aws_tmp_files()


    # ###############################
    # # extract zip and put in /tmp/
    # ###############################
    # """
    # using a zipped model is optional.
    # using a zipped model is possible but may merely add
    # time and labor to the Lambda-function
    # """

    # # name path of zip archive
    # archive_file_path = "/tmp/modelzip.zip"
    # destination_file_path = "/tmp/"
      
    # # open zip archive (read mode)
    # with ZipFile(archive_file_path, 'r') as zip: 

    #     # extract files to destination_file_path
    #     zip.extractall( destination_file_path )

    # # for terminal: see what files exist in /tmp/
    # print_aws_tmp_files()

    ##############
    # Load Model
    ##############

    try: 
        # Set model path (including directory)
        model_path = "/tmp/" + s3_file_name

        # set up TF interpreter (point at .tflite model)
        interpreter = Interpreter(model_path)

        # for terminal
        print("Model Loaded Successfully.")


    except Exception as e:

        output = f"""Error: Could not load model. Path = {model_path}
        Error Message = {str(e)} 
        """
        
        # print for terminal
        print(output)

        statusCode = 403

        # End the lambda function
        return {
            'statusCode': statusCode,
            'body': output
        }


    ###############
    # Set up Model
    ###############

    # set up interpreter
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # X: input data
    X_raw_input = [[user_input_for_X]]

    ## for testing
    #X_raw_input = [[0.37094948]]

    # formatting: convert raw input number to an numpy array
    input_data = np.asarray(X_raw_input, dtype=np.float32)
        
    # y: using model, produce predicted y from X input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Start interpreter
    interpreter.invoke()


    ##################
    # Make Prediction 
    ##################

    # Make Prediction
    tflite_prediction_results = interpreter.get_tensor(output_details[0]['index'])

    # for terminal
    print("Prediction: y =", tflite_prediction_results)


   ##############################
    # Final Clean Up Lambda /tmp/
    ##############################
    # Clear AWS Lambda Function /tmp/ directory
    clear_tmp_directory()


    ############################
    # process and format output
    ############################
    """
    - remove brackets (remove from matrix/array), isolate just the number
    - make type -> float
    """
    tflite_prediction_results = tflite_prediction_results[0]
    tflite_prediction_results = float( tflite_prediction_results[0] )


    ###############
    # Final Output
    ###############
    status_code = 200
    output = tflite_prediction_results

    return {
        'statusCode': status_code,
        'about': "output_of_Tensorflow_ML_model",
        'body': output
    }
