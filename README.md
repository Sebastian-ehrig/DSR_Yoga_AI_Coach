Earthquake_predictor
==============================

Predicting the level of damage to buildings caused by the Gorkha earthquake depending on building location and building construction. The Data that is used for the model predictions are based on surveys conducted by Kathmandu Living Labs and the Central Bureau of Statistics.

This model was developed during the DSR (DataScienceRetreat) Batch30.


Model workflow and data-analysis
-----------------------------------------

1. Loading the raw data (training-data, test-data, labels) 
2. Concatenate training- and test-data; 
   Clean data e.g. convert str to int where applicable
3. Encode data: reads in the values and the labels, applies target encoding and 
   returns a dataframe which consists only of numeric columns.
4. Model training: defin ML model and grid of attributes. Fit the model to the train values of each attribute set.
   then choose best model as fitted model according to f1(micro) score.
5. Compute predictions and generate the reports.


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
