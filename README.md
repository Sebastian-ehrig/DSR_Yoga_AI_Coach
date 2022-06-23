DSR_Yoga_AI_Coach
=================

The Yoga_AI_Coach is a python based application able to detect and correct yoga-poses to lead a Yoga training sequence composed of different Yoga Asanas (Yoga poses). The application is able to detect in real-time all body movements captured via a camera using TensorflowLite Movenet for pose detection.

A key feature of the application is that it is able to lead a yoga-session through voice commands:
As soon as the practicing person gets into the correct pose that was instructed by the voice command, the App will provide suggestions to correct the Asana. Once the practicing person has settled into the correct Asana, the voice command continues with further instructions until the training sequence is completed.

Further technical notes:
------------------------

The correct keypoint-positions of specific Asanas are pre-determined from still-images. A neural network was trained on key Asanas to be able to make predictions on multiple yoga poses.

This model was developed during the DSR (DataScienceRetreat) Batch30.

Model workflow and data-analysis
--------------------------------

Data notations used in the scope of this project:
-------------------------------------------------
