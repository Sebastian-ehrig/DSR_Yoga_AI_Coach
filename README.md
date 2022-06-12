DSR_Yoga_AI_Coach
==============================

The Yoga_AI_Coach is a python based application able to detect and correct yoga-poses to lead a Yoga training sequence composed of different Yoga Asanas (Yoga poses). The application is able to detect in real-time all body movements captured via a camera using TensorflowLite Movenet for pose detection.

A key feature of the application is that it is able to lead a yoga-session through voice commands:
As soon as the practicing person gets into the correct pose that was instructed by the voice command, the App will provide suggestions to correct the Asana. Once the practicing person has settled into the correct Asana, the voice command continues with further instructions until the training sequence is completed.

Further technical notes:
-----------------------------------------

The correct keypoint-positions of specific Asanas are pre-determined from still-images. A neural network was trained on key Asanas to be able to make predictions on multiple yoga poses. 

This model was developed during the DSR (DataScienceRetreat) Batch30.

Model workflow and data-analysis
-----------------------------------------




Project Organization and folder structure
-----------------------------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Models generated.  
    │   └── Model adjustments    <- Contains log-files for keeping track of model refinements.
    │
    ├── reports            <- Contains reports in html format as well as csv-files keeping track 
    │                         of all the analysis that have been perfomed.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── functions      <- Scripts to turn raw data into features for modeling
    │   │   │                 
    │   │   ├── encoder.py        <- Function to encode the raw data.
    │   │   ├── helper.py         <- Functions for loading and conversion of the data.
    │   │   ├── test.py           <- Contains function for testing the model.
    │   │   └── train_model.py    <- Function for training the model.
    │   │   
    │   ├── helper             <- Scripts to train models and then use trained models to make 
    │   │   │                     predictions
    │   │   └── conf.py        <- Contains all file-paths and data-keys
    │   │
    │   ├── main.py            <- Script to execute the entire analysis and model predicion
    │   └── train.py           <- Script for encoding the data, taining the model and making 
    │                             the final predictions
    │
    ├── test_environment.py   <- Test the dvelopment environment to make sure all packages are
    │                            working properly.
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
Data notations used in the scope of this project:
-----------------------------------------

raw data: 
 train_values 
 train_labels 
 test_values
 X_all_raw (train & test values)

preprocessed data:
 X_train
 y_train
 X_test

modeled data:
 fitted_model (cv on train set)

output data:
 prediction (on test set)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
