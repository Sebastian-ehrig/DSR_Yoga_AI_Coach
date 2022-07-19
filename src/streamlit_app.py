import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 #cv2
import warnings
import glob
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

from helper.conf import *
from functions.helper import *
from functions.crop_algorithm import *
from functions.pose_calc import *
from functions.corrections_streamlit import *
from functions.sequence_lead import *

st.header("YogaAI Demo")
st.subheader("Real time pose-estimation and pose-correction using TensorFlow")

# check what cameras are available
cams = glob.glob("/dev/video?")
print("Available cameras:", cams)

#------------
# Load Model:
#------------

confidence_score = 0.2 # threshold for drawing the keypoints

play_yoga_sequence = 0 # 1 if yoga sequence is to be played

# enhance_contrast = 0 # if contrast enhancement is to be done

# Load Model:
tensorflow_model = st.radio(
     "Select TensorFlow model",
     ('Movenet_singlepose_thunder', 'Movenet_singlepose_lightning'))

    # Model: Movenet singlepose thunder3
if tensorflow_model=='Movenet_singlepose_thunder3':
    interpreter = tf.lite.Interpreter(model_path=model_path_thunder_3)
    interpreter.allocate_tensors()
    image_size=(256, 256)

    # Model: Movenet singlepose lightning3
else:
    interpreter = tf.lite.Interpreter(model_path=model_path_lightning_3)
    interpreter.allocate_tensors()
    image_size=(192, 192)

# Slider for selecting the threshold for drawing the keypoints
# DEFAULT_CONFIDENCE_THRESHOLD = 0.2
# confidence_score = st.slider(
#         "Confidence threshold for detecting the keypoints", 0.0, 0.6, DEFAULT_CONFIDENCE_THRESHOLD, 0.01)

# ignore warnings
warnings.filterwarnings('ignore')

# Variables to calculate FPS
counter, fps = 0, 0
startTime = time.time()
CommandExecuted = False

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

    
seq_step = 0 # sequence step

def video_frame_callback(input_image):

    frame = cv2.flip(input_image, 1)
    
    # input frame has to be a float32 tensor.
    # RGB with values in [0, 255].
    image_height, image_width, _ = frame.shape

    # # make keypoint predictions on image frame
    # ------------------------------------------

    # improve keypoint-predictions by cropping the image around the intitaly detected keypoints
    crop_region = init_crop_region(image_height, image_width)
    keypoints_with_scores = improve_predictions(make_predictions_compact, frame, crop_region, image_size, interpreter)

    # Render image including the detected keypoints:
    # ----------------------------------------------
    confidence_threshold = confidence_score # threshold for drawing the keypoints

    # draw the line connections
    frame = draw_connections(frame, keypoints_with_scores, EDGES, confidence_threshold)
    
    # draw the keypoints
    # draw_keypoints_initial(frame, initial_keypoints_with_scores, confidence_threshold)
    frame = draw_keypoints(frame, keypoints_with_scores, confidence_threshold)

    # draw_angles(frame, keypoints_with_scores, confidence_threshold)

    # run the classifier on the frame
    prob_list_labels, prob_list_scores, output, labels = classifier(keypoints_with_scores)

    # Compute cosine-similarity score using the keypoints:
    # -----------------------------------------------------
    # get index of closest matching pose
    pose_idx = np.where(output == np.amax(output))[0][0]

    keypoints_reference_pose = poses_df[pose_idx]
    pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
    pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

    cos_sim_score_kpt = cosine_similarity(
                            pose_angles_reference_img,
                            pose_angles_current_frame
                    )

    mse = (np.square(pose_angles_reference_img - pose_angles_current_frame)).mean()

    # draw class prediction results
    frame = draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame)
    # draw cosine-similarity scores
    frame = draw_prediction_scores(keypoints_with_scores, cos_sim_score_kpt, mse, frame)

    
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
                    correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)


        if seq_step == 1:
            if time.time() - time_in > 90:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 2:
            if time.time() - time_in > 50:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 3:
            if time.time() - time_in > 45:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)

        if seq_step == 4:
            if time.time() - time_in > 95:
                if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                    if mse <= 200:
                        correct = True
                    if mse > 201:              
                        correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)

    else: # corrections for all poses
        if counter % 50 == 0: # suggest corrections every 50 frames (~ 2 seconds)
                if mse <= 150:
                    correct = True
                    CorrectPose = "./src/functions/sequence_commands/Correct.ogg"
                    playSound_ST(CorrectPose)
                if mse > 201:              
                    correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx)
        
    return frame


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = video_frame_callback(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit component which deals with video and audio real-time I/O through web browsers
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {
            "width": {"min": 800, "ideal": 1200, "max": 1920 },
        }, "audio": False}, 
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

st._transparent_write("Pose-predictions are performed once the person is fully visible in the camera")