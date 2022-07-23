import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings
import tqdm
import glob
import time

from helper.conf import *
from functions.helper import *
from functions.crop_algorithm import *
from functions.pose_calc import *
from functions.corrections import *
from functions.sequence_lead import *

# check what cameras are available
cams = glob.glob("/dev/video?")
print("Available cameras:", cams)

# ignore warnings
warnings.filterwarnings('ignore')

#-------------------
# Define parameters:
#-------------------

single_pose_thunder3=1 # Movenet singlepose thunder3 to be used, 
                       # else Movenet singlepose lightning3
capture_frames=0 # if frames are to be captured

confidence_score = 0.2 # threshold for drawing the keypoints

external_cam = 0 # 1 if external camera is to be used

play_yoga_sequence = 0 # 1 if yoga sequence is to be played

# enhance_contrast = 0 # if contrast enhancement is to be done

high_res = 0 # 1 if HD resolution is to be used

seq_step = 0 # sequence step 

full_screen_mode = 1 # 1 if full screen mode is to be used

# Variables to calculate FPS
counter, fps = 0, 0
startTime = time.time()

# ----------------------
# Load TensorFlow model:
# ----------------------

    # Model: Movenet singlepose thunder3
if single_pose_thunder3==1:
    interpreter = tf.lite.Interpreter(model_path=model_path_thunder_3)
    interpreter.allocate_tensors()
    image_size=(256, 256)

    # Model: Movenet singlepose lightning3
else:
    interpreter = tf.lite.Interpreter(model_path=model_path_lightning_3)
    interpreter.allocate_tensors()
    image_size=(192, 192)

#---------------------------------------------
# initialize video frame capture using OpenCV2
#---------------------------------------------

# VideoCapture(0) -> webcam
# VideoCapture(2) -> external cam/webcam

if external_cam==1: # external cam
    cap = cv2.VideoCapture(2)
else:
    cap = cv2.VideoCapture(0) # webcam

# set camera resolution etc.
if high_res == 1:
# # 1080p
    frame_Width = 1920
    frame_Height= 1080
else:
# # 720p
    frame_Width = 1280
    frame_Height= 720

cap.set(3,frame_Width) # Width of the frames in the video stream
cap.set(4,frame_Height) # Height of the frames in the video stream

# ---------------------
# Load reference poses
# ---------------------

poses_df = []
ref_images = []
for pose_idx in range(10):

    ref_image_path = './reference_poses/Yoga_Seq_'+str(pose_idx) +'.jpg'
    pose_df = pd.read_csv('./reference_poses/Yoga_Seq_'+str(pose_idx) +'.csv', sep='\t')
    pose_df = pose_df.to_numpy()
    pose_df = np.squeeze(pose_df)
    poses_df.append(pose_df)

# ---------------------------------------
# Load reference image and find contours:
# ---------------------------------------

contour_image_path = './reference_poses/contour/tadasana' +'.jpg'
contour_image = tf.io.read_file(contour_image_path)
contour_image = tf.image.decode_jpeg(contour_image)
contour_image = tf.image.resize_with_pad(np.expand_dims(contour_image, axis=0), frame_Height, frame_Width)
contour_image = np.squeeze(contour_image.numpy(), axis=0)
contour_image = contour_image.astype(np.uint8)
# find image contour
img_grey = cv2.cvtColor(contour_image,cv2.COLOR_BGR2GRAY)
thresh = 150
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ---------------------------------------------------------------------
# *** Begin frame capture ***
# ---------------------------------------------------------------------

# Full screen mode
if full_screen_mode == 1:
    WINDOW_NAME = 'Full_Screen_Mode'
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while cap.isOpened():

    # read frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    counter += 1 # frame counter

    # input frame has to be a float32 tensor of shape: 256x256x3 or 196x196x3.
    # RGB with values in [0, 255].
    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_height, image_width, _ = frame.shape

    # # make keypoint predictions on image frame
    # initial_keypoints_with_scores = make_predictions(img, interpreter, image_size)

    # improve keypoint-predictions by cropping the image around the intitaly detected keypoints
    crop_region = init_crop_region(image_height, image_width)
    keypoints_with_scores = improve_predictions(make_predictions_compact, img, crop_region, image_size, interpreter)

    # draw contour
    # ------------------
    draw_reference_contour(contours, frame, keypoints_with_scores)

    # Render image including the detected keypoints:
    # ----------------------------------------------
    confidence_threshold = confidence_score # threshold for drawing the keypoints

    # draw the line connections
    draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
    
    # draw the keypoints
    # draw_keypoints_initial(frame, initial_keypoints_with_scores, confidence_threshold)
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    draw_rectangle_around_keypoints(frame, keypoints_with_scores)

    # draw_angles(frame, keypoints_with_scores, confidence_threshold)

    # run the classifier on the frame
    prob_list_labels, prob_list_scores, output, labels = classifier(keypoints_with_scores)

    # get index of closest matching pose
    pose_idx = np.where(output == np.amax(output))[0][0]

    # get the keypoints of the closest matching pose
    keypoints_reference_pose = poses_df[pose_idx]

    # calculate angles for the keypoints of the reference pose and the keypoints of the current pose
    pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
    pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

    # compute cosine-similarity score
    cos_sim_score_kpt = cosine_similarity(pose_angles_reference_img, pose_angles_current_frame)

    # compute mean-squared error
    mse = (np.square(pose_angles_reference_img - pose_angles_current_frame)).mean()
    mse = nan_to_integer(mse)
    # draw class prediction results
    draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame)

    # draw cosine-similarity scores
    draw_prediction_scores(keypoints_with_scores, cos_sim_score_kpt, mse, frame)

    if play_yoga_sequence == 1:
        seq_step, time_in = yoga_sequence_lead(keypoints_reference_pose, keypoints_with_scores, pose_idx, seq_step, mse)

    # # --------------------------------
    # # make pose-correction suggestions
    # # --------------------------------

    if play_yoga_sequence == 1: # corrections for yoga sequence

        if seq_step == 0 or seq_step >=5:
            if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                if mse <= 200:
                    correct = True
                if mse > 201:              
                    correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 1:
            if time.time() - time_in > 90:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 2:
            if time.time() - time_in > 50:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 3:
            if time.time() - time_in > 45:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 4:
            if time.time() - time_in > 95:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)

    else: # corrections for all poses
        if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                if mse <= 150:
                    correct = True
                    CorrectPose = "./src/functions/sequence_commands/Correct.ogg"
                    playSound(CorrectPose)
                if mse > 201:              
                    correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)
        
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if full_screen_mode == 1:
        cv2.imshow(WINDOW_NAME, frame)
    else:
        cv2.imshow('MoveNet frame', frame)

    if capture_frames == 1:
        cv2.imwrite('./frames/Frame'+str(counter)+'.jpg', frame)
    
    # define brake-out: if we hit "q" on the keyboard
    # frame capure is stopped
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release() # release the camera
cv2.destroyAllWindows() # close all windows

# save the last image frame
cv2.imwrite('./frames/Frame'+str(0)+'.jpg', frame)

