# code for a local TFlite (Tensor Flow Lite) test, assuming an env where tflite is installed, see instructions below

"""
sample zipped test env should be available here:
https://github.com/lineality/tensorflow_lite_in_aws_lambda_function

to use pre-made-zipped tflite env for python 3.8:

$ unzip env.zip
$ source env/bin/activate
(env) $ python3 local_tflite_test.py

Note: you can install python3.8 separately if your default is something else.

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
import numpy as np  # not needed?

import glob  # for directory file search
#from zipfile import ZipFile  # Optional


def run_model():


    # set up TF interpreter (point at .tflite model)
    interpreter = Interpreter('model.tflite')

    # for terminal
    print("Model Loaded Successfully.")



    ###############
    # Set up Model
    ###############

    # set up interpreter
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # X: input data
    X_raw_input = [[0.37094948,0.37094948,0.37094948,0.37094948]]

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

run_model()

