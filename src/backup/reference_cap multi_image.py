import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import warnings
import glob
import random

from helper.conf import *
from functions.helper import *

# check what cameras are available
cams = glob.glob("/dev/video?")
# cams[2]

#------------
# Load Model:
#------------

single_pose_thunder3=1 # Movenet singlepose thunder3 to be used, 
                       # else Movenet singlepose lightning3

capture_frames=1 # if frames are to be captured

confidence_score = 0.3 # threshold for making the decision

    # Model: Movenet singlepose thunder3
if single_pose_thunder3==1:
    interpreter = tf.lite.Interpreter(model_path=model_path_thunder_3)
    interpreter.allocate_tensors()
    image_size=(256, 256)
    input_size = 256


    # Model: Movenet singlepose lightning3
else:
    interpreter = tf.lite.Interpreter(model_path=model_path_lightning_3)
    interpreter.allocate_tensors()
    image_size=(192, 192)
    input_size = 192



# ignore warnings
warnings.filterwarnings('ignore')

#---------------------------------------------
# initialize video frame capture using OpenCV2
#---------------------------------------------

# path where the image sequence is stored
# paths = glob.glob("./Yoga_sequences/Yoga_Flow_Printable_7.4.14/*")
paths = glob.glob("./frames/*")

sorted_paths = sorted(paths)

for i in np.arange(len(sorted_paths)):
    image_path = sorted_paths[i]

#---------------------------------------------
# Load single input image
#---------------------------------------------

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    h, w, c = image.shape
    if c<3:
        image = np.stack((image,)*3, axis=-1).squeeze()
    #save directory
    file_dir = './reference_frames/'
        
        # input frame has to be a float32 tensor of shape: 256x256x3.
        # RGB with values in [0, 255].
        
        # resize image to fit the expected size

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    # input_image = tf.expand_dims(image, axis=0)
    # input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    img = tf.image.resize(np.expand_dims(image, axis=0), image_size)
        # convert to float32
    input_image = tf.cast(img, dtype=tf.float32)

        # SETUP INPUT and OUTPUT
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
        
        # Make predictions:
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    confidence_threshold = confidence_score # threshold for making the decision

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)

    frame = np.squeeze(display_image.numpy(), axis=0)

        # draw the line connections
    draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
        
        # draw the keypoints
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    draw_angles(frame, keypoints_with_scores, confidence_threshold)

        # run the classifier on the frame
    prob_list_labels, prob_list_scores = classifier(keypoints_with_scores)

    draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame)


    plt.imshow(frame)

    save_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    if capture_frames == 1:
        if not cv2.imwrite('./reference_frames/'+'Yoga_Seq_'+ str(i) + '.jpg', save_frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
            raise Exception("Could not write image")
