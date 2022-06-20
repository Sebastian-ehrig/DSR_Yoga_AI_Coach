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
from functions.crop_algorithm import *
from functions.pose_calc import *
from functions.corrections import *

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

# enhance_contrast = 0 # if contrast enhancement is to be done

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

cap = cv2.VideoCapture(0)

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
# frame_Width = 1920
# frame_Height= 1080
# cap.set(3, 1920)
# cap.set(4, 1080)
# # 720p
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(3,frame_Width) # Width of the frames in the video stream
# cap.set(4,frame_Height) # Height of the frames in the video stream
# cap.set(5, 30) # Frame rate

# # define a white frame
# white_frame = np.zeros([frame_Height,frame_Width,3],dtype=np.uint8)
# white_frame.fill(255)
# # or img[:] = 255

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

num=0

while cap.isOpened():

    # read frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    
    counter += 1

    # input frame has to be a float32 tensor of shape: 256x256x3.
    # RGB with values in [0, 255].
    img = frame.copy()   
    image_height, image_width, _ = frame.shape

    # # make keypoint predictions on image frame
    # initial_keypoints_with_scores = make_predictions(img, interpreter, image_size)

    # # determine crop region
    # determine_crop_region(keypoints_with_scores, image_height, image_width)
    
    # improve keypoint-predictions by cropping the image around the intitaly detected keypoints
    crop_region = init_crop_region(image_height, image_width)
    keypoints_with_scores = improve_predictions(make_predictions_compact, img, crop_region, image_size, interpreter)

    # Render image including the detected keypoints:
    # ----------------------------------------------
    
    confidence_threshold = confidence_score # threshold for making the decision

    # draw the line connections
    draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
    
    # draw the keypoints
    # draw_keypoints_initial(frame, initial_keypoints_with_scores, confidence_threshold)
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    # draw_angles(frame, keypoints_with_scores, confidence_threshold)

    # run the classifier on the frame
    prob_list_labels, prob_list_scores, output, labels = classifier(keypoints_with_scores)

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
    # cos_sim_score_kpt = cosine_similarity(
    #     np.array(reference_pose_angles(poses_df[pose_idx])),
    #     np.array(pose_angles(keypoints_with_scores))
    #     )
    keypoints_reference_pose = poses_df[pose_idx]
    pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
    pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

    cos_sim_score_kpt = cosine_similarity(
                            pose_angles_reference_img,
                            pose_angles_current_frame
                    )

    mse = (np.square(pose_angles_reference_img - pose_angles_current_frame)).mean()

    # draw class prediction results
    draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame)
    # draw cosine-similarity scores
    draw_cosine_similarity(keypoints_with_scores, cos_sim_score_kpt, mse, frame)


    correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx)


    # draw_FPS(frame, counter, fps, start_time) 
       
    cv2.imshow('MoveNet frame', frame)

    if capture_frames == 1:
        num += 1 
        cv2.imwrite('./frames/Frame'+str(num)+'.jpg', frame)
    
    # define brake-out: if we hit "q" on the keyboard
    # frame capure is stopped
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
        
cap.release() # release the camera
cv2.destroyAllWindows() # close all windows


## show the last captured frame
webcam_frame = frame.copy()
plt.imshow(webcam_frame)
# print shape
print(f'Image shape is: {webcam_frame.shape}')

# save the last image frame
cv2.imwrite('Frame'+str(random.randint(1, 1000_000))+'.jpg', frame)
cv2.imwrite('Frame'+str(num)+'.jpg', frame)

