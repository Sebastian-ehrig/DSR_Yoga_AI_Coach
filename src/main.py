import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings
import glob
import random
import tqdm
import time

# for computing cosine similarity
from img2vec_pytorch import Img2Vec
from PIL import Image
# from playsound import playsound

from helper.conf import *
from functions.helper import *
from functions.corrections import *
from functions.pose_calc import *

# check what cameras are available
cams = glob.glob("/dev/video?")
# cams[2]

#------------
# Load Model:
#------------

single_pose_thunder3=1 # Movenet singlepose thunder3 to be used, 
                       # else Movenet singlepose lightning3

capture_frames=0 # if frames are to be captured

confidence_score = 0.2 # threshold for making the decision

check_cam_resolution=0 # if camera resolution is to be checked

enhance_contrast = 0 # if contrast enhancement is to be done

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

# ignore warnings
warnings.filterwarnings('ignore')

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

#---------------------------------------------
# initialize video frame capture using OpenCV2
#---------------------------------------------

# VideoCapture(0) -> webcam
# VideoCapture(2) -> external cam/webcam

cap = cv2.VideoCapture(2)

# check camera resolution

# if check_cam_resolution == 1:

#     # check available camera resolutions (requires lxml)
#     url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
#     table = pd.read_html(url)[0]
#     table.columns = table.columns.droplevel()
#     cap = cv2.VideoCapture(0)
#     resolutions = {}
#     for index, row in table[["W", "H"]].iterrows():
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#         width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         resolutions[str(width)+"x"+str(height)] = "OK"
#     print(resolutions)




# codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
# cap.set(6, codec)
# cap.set(5, 30)
# cap.set(3, 1920)
# cap.set(4, 1080)

# set camera resolution etc.
# # 1080p
frame_Width = 1920
frame_Height= 1080
# # 720p
# cap.set(3, 1280)
# cap.set(4, 720)
cap.set(3,frame_Width) # Width of the frames in the video stream
cap.set(4,frame_Height) # Height of the frames in the video stream
# cap.set(5, 30) # Frame rate

# define a white frame to be used for the overlay
white_frame = np.zeros([frame_Height,frame_Width,3],dtype=np.uint8)
white_frame.fill(255)

num=0

# Load reference pose for computing the cosine-similarity
# -------------------------------------------------------
poses_df = []
ref_images = []
for pose_idx in range(5):

    ref_image_path = './reference_poses/Yoga_Seq_'+str(pose_idx) +'.jpg'
    pose_df = pd.read_csv('./reference_poses/Yoga_Seq_'+str(pose_idx) +'.csv', sep='\t')
    pose_df = pose_df.to_numpy()
    pose_df = np.squeeze(pose_df)
    # pose_df = pose_df.ravel()   
    poses_df.append(pose_df)

    #Initialize Img2Vec without GPU
    img2vec = Img2Vec(cuda=False)

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_image_pil = Image.fromarray(ref_image)
    ref_img  =  img2vec.get_vec(ref_image_pil)

    ref_images.append(ref_img)

# cos_score = cosine_sim(poses_df[0],poses_df[4])
# average_cos_sim = sum(cos_score)/len(cos_score)
# print(average_cos_sim)

while cap.isOpened():

    # read frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    # input frame has to be a float32 tensor of shape: 256x256x3.
    # RGB with values in [0, 255].

    counter += 1
    
    img = frame.copy()   

    # ----------------------------------------------------------------------
    # enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # and LAB Color space https://www.xrite.com/blog/lab-color-space
    # ----------------------------------------------------------------------
    if enhance_contrast == 1:
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0] # L*: Lightness
        a_channel = lab[:,:,1] # A*: red-green
        b_channel = lab[:,:,2] # B*: blue-yellow

        # Applying CLAHE to L-channel:
        # maybe try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a- and b-channel
        limg = cv2.merge((cl,a_channel,b_channel))

        # Converting image from LAB Color model to BGR color spcae
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # ----------------------------------------------------------------------
    
    # resize image to fit the expected size
    img = tf.image.resize(np.expand_dims(img, axis=0), image_size)
    # convert to float32
    input_image = tf.cast(img, dtype=tf.float32)

    # SETUP INPUT and OUTPUT
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions:
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # OUTPUT ("keypoints_with_scores") is a float32 tensor of shape [1, 1, 17, 3].
    
    # The FIRST TWO channels of the last dimension represents 
    # the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) 
    # of the 17 keypoints (in the order of: [nose, left eye, right eye, left ear, 
    # right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, 
    # right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle])
    
    # e.g. right_eye = keypoints_with_scores[0][0][2]
    # Note: the exact pixel position in this example is: np.array(right_eye[:2]*[frame_Height,frame_Width]).astype(int)
    # --> np.array(keypoints_with_scores[0][0][2][:2]*[frame_Height,frame_Width])

    # The THIRD channel of the last dimension represents the PREDICTION CONFIDENCE SCORES 
    # of each keypoint, also in the range [0.0, 1.0]   
    
    # Render image including the detected keypoints:
    # ----------------------------------------------
    
    confidence_threshold = confidence_score # threshold for making the decision

    # draw the line connections
    draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
    
    # draw the keypoints
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    # draw_angles(frame, keypoints_with_scores, confidence_threshold)

    prob_list_labels, prob_list_scores, output, labels = classifier(keypoints_with_scores)


    # run the classifier on the frame
    # correct = False 
    # while not correct: 
    #      # at the beginning of each Aanas, apply classifier
    #     prob_list_labels, prob_list_scores, output, labels = classifier(keypoints_with_scores)
    #     if prob_list_scores[0] >= 0.5: 
    #         correct = True
    #     else:
    #         play.correction

    # Compute cosine-similarity score using the keypoints:
    # -----------------------------------------------------
    # get index of closest matching pose
    pose_idx = np.where(output == np.amax(output))[0][0]

    # current_keypoint_coordinates = keypoints_with_scores[0][0][:,0:2]
    # current_keypoint_coordinates = np.squeeze(current_keypoint_coordinates)
    # cos_score = cosine_sim(poses_df[1], current_keypoint_coordinates)
    # cos_sim_score_kpt = sum(cos_score)/len(cos_score)

    # cosine similarity from angle differences
    # ----------------------------------------
    cos_sim_score_kpt = cosine_similarity(
        np.array(reference_pose_angles(poses_df[pose_idx])),
        np.array(pose_angles(keypoints_with_scores))
        )

    # print(cos_sim_score_kpt)
    # Compute cosine-similarity score between frame and reference frame
    # -----------------------------------------------------
    # # draw the line connections
    # draw_connections(white_frame, keypoints_with_scores, EDGES, confidence_threshold)
    # # draw the keypoints
    # draw_keypoints(white_frame, keypoints_with_scores, confidence_threshold)
    # white_frame_pil = Image.fromarray(white_frame)
    # current_frame  =  img2vec.get_vec(white_frame_pil)
    # cos_sim_score_frame = cosine_similarity(ref_img, current_frame)
    # ----------------------------------------------------------------------


    # if output[0] >= 0.9: 
    #         correct = True
    # else:
    #         play.correction      

    # check the probability in the prob_list_scores
    # if score < 0.6 Play.correction
    # if the score stays low then play.sleep
    # if score > play.Asanapose

    # correct_angles(frame, keypoints_with_scores, confidence_threshold)

    # draw class prediction results
    draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame)
    # draw cosine-similarity scores
    draw_cosine_similarity(keypoints_with_scores, cos_sim_score_kpt, frame)

    # draw_FPS(frame, counter, fps, start_time)

    cv2.imshow('MoveNet singlepose', frame)

    if capture_frames == 1:
        num += 1 
        cv2.imwrite('./frames/Frame'+str(num)+'.jpg', frame)
    
    # define brake-out: if we hit "q" on the keyboard
    # frame capure is stopped
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
        
cap.release() # release the camera
cv2.destroyAllWindows() # close all windows


# ## show the last captured frame
# webcam_frame = frame.copy()
# plt.imshow(webcam_frame)
# # print shape
# print(f'Image shape is: {webcam_frame.shape}')

# # save the last image frame
cv2.imwrite('Frame'+str(random.randint(1, 1000_000))+'.jpg', frame)
cv2.imwrite('Frame'+str(num)+'.jpg', frame)

