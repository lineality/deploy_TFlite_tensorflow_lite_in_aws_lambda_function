# code for AWS-Lambda-function to deploy model.tflite in AWS 
# for CV computer vision, using PIL/Pillow for preprocessing

"""
# See:
https://github.com/lineality/deploy_TFlite_tensorflow_lite_in_aws_lambda_function

# To install the tensorflow lite runtime package follow official tensorfow docs:
https://www.tensorflow.org/lite/guide/python


# instruction code to create python env (for uploading to AWS):
# only tflite_runtime is needed, numpy is included with tflite

$ python3 -m venv env; source env/bin/activate

$ pip3 install --upgrade pip

# version 2.8 may cause AWS errors, try 2.7
# use: pip install tflite-runtime==2.7.0

$ pip3 install pillow

$ pip3 freeze > requirements.txt


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
# edit file in the zipped archive
# or re-add a new .py to replace the old by repeating the same step from above
$ zip -g ./function.zip -r lambda_function.py
"""



"""
Workflow:

1. get user_input
2. S3: Connect to S3 (Make resource and client)
3. download model.tflite file from S3
4. download image file
5. PIL/Pillow: open image file (w/ Python Image Library) to specs
6. load model
7. make prediction
8. clear /tmp/
9. export result
"""

"""
Sample Ouput:
{
  "statusCode": 200,
  "about": "output_of_Tensorflow_ML_model",
  "body": ?
}
"""

"""
Sample input: 
{
  "s3_file_path_AI_model": "FOLDER_NAME/FOLDER_NAME/model.tflite",
  "s3_file_path_picture_file": "YOUR_FOLDER_NAME/PIC_NAME.jpeg",
  "s3_bucket_name": "YOUR_AWS_S3_BUCKET_NAME"
}
"""

# import librarires
import boto3            # for AWS
import glob             # for directory file search
import json
import numpy as np      # for input and output processing
from PIL import Image   # for image processing
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter 


###################
# Helper Functions
###################

# Helper Function
def get_file_from_S3_to_lambda_tmp(s3_resource, s3_bucket_name, s3_file_path, lambda_tmp_file_name):

    # s3_resource.meta.client.download_file('YOUR_BUCKET_NAME', 'FILE_NAME.txt', '/tmp/FILE_NAME.txt')
    s3_resource.meta.client.download_file( s3_bucket_name, s3_file_path, lambda_tmp_file_name )
    
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


################
# Main Function
################
def lambda_handler(event, context):

    #################
    # Get User Input
    #################

    # get s3_file_path_AI_model and path in s3
    # Test if input exists and can be processed
    try:
        s3_file_path_AI_model = event["s3_file_path_AI_model"]
       
        # terminal
        print( s3_file_path_AI_model )


        # slice out just the name of the model from the whole path
        S3_file_name_AI_model = s3_file_path_AI_model.split('/')[-1]

    except Exception as e:
 
        output = f"""Error: No input for s3_file_path_AI_model 
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
    # Test if input exists and can be processed
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



    # get s3_file_path_picture_file in s3
    # Test if input exists and can be processed
    try:
        s3_file_path_picture_file = event["s3_file_path_picture_file"]

        # slice out just the name of the model from the whole path
        S3_file_name_picture_file = s3_file_path_picture_file.split('/')[-1]

    except Exception as e:
 
        output = f"""Error: No input for s3_file_path_picture_file 
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


    ##########################################
    # load files from S3 int AWS-python-/tmp/
    ##########################################

    # AWS Files Name
    model_aws_tmp_file_name = "/tmp/" + S3_file_name_AI_model
    picture_file_aws_tmp_file_name = "/tmp/" + S3_file_name_picture_file

    try:
        ############################
        # Get AI Model file from S3
        ############################
        """
        (Docs)
        get_file_from_S3_to_lambda_tmp(s3_resource, 
                                       s3_bucket_name, 
                                       s3_file_path, 
                                       lambda_tmp_file_name
                                       )
        """

        # Get AI Model

        get_file_from_S3_to_lambda_tmp(s3_resource, 
                                       s3_bucket_name, 
                                       s3_file_path_AI_model, 
                                       model_aws_tmp_file_name)

    except Exception as e:

        output = f"""Error: Could not get AI Model file from S3
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


    try:
        ###########################
        # Get Picture file from S3
        ###########################

        # Get Picture File
        get_file_from_S3_to_lambda_tmp(s3_resource, 
                                       s3_bucket_name, 
                                       s3_file_path_picture_file, 
                                       picture_file_aws_tmp_file_name)

    except Exception as e:

        output = f"""Error: Could not get Picture file from S3
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


    # for terminal: see what files exist in aws /tmp/
    print_aws_tmp_files()



    ##############
    # Load Model
    ##############

    try: 
        # Set model path (including directory)
        model_path = "/tmp/" + S3_file_name_AI_model

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

    # select test image
    # img_path = 'PIC_NAME_HERE.1.jpg'
    img_path = picture_file_aws_tmp_file_name
    
    
    #################################
    # PIL/Pillow image preprocessing 
    #################################

    # load and resize image file
    """
    equivilent of this from Keras:

    img = image.load_img(img_path, target_size=(224, 224))
    """
    img = Image.open(img_path)
    img = img.resize((224, 224))

    # image -> array
    """
    equivilent of this from Keras:

    img_array = image.img_to_array(img)
    """
    img_array = np.asarray(img)

    # already numpy
    expanded_img_array = np.expand_dims(img_array, axis=0)

    # already numpy
    preprocessed_img = expanded_img_array / 255. 

    # set: input_data = preprocessed image
    input_data = preprocessed_img

    # type cast to float32
    input_data = input_data.astype('float32')


    #######################
    # End of preprocessing
    #######################
        
    # y: using model, produce predicted y from X input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Start interpreter
    interpreter.invoke()


    ##################
    # Make Prediction 
    ##################

    # Make Prediction
    tflite_prediction_results = interpreter.get_tensor(output_details[0]['index'])


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
    # for terminal
    print("1 Prediction: y =", tflite_prediction_results)

    tflite_prediction_results = tflite_prediction_results[0]

    # for terminal
    print("2 Prediction: y =", tflite_prediction_results)

    # reformat results: turn into string form of just two numbers
    tflite_prediction_results = str(tflite_prediction_results)
    tflite_prediction_results = tflite_prediction_results.replace("[", "")
    tflite_prediction_results = tflite_prediction_results.replace("]", "")
    tflite_prediction_results =  tflite_prediction_results.split("  ")
    
    # for terminal
    print("3 Prediction: y =", tflite_prediction_results)

    # get second probability: probability of damage
    tflite_prediction_results = tflite_prediction_results[1]
    
    # for terminal
    print("4 Prediction: y =", type(tflite_prediction_results), tflite_prediction_results )

    #tflite_prediction_results = float(tflite_prediction_results)


    ###############
    # Final Output
    ###############
    status_code = 200
    output = tflite_prediction_results

    return {
        'statusCode': status_code,
        'about': """Probability of Damage
                    Output of Tensor Flow Keras Transfer Learning 
                    Computer Vision Neural Network Deep Learning Model 
                    run on TFlite in a compact python 3.8 venv
                    with Python Image Libarary input image preprocessing""",
        'body': output
    }
