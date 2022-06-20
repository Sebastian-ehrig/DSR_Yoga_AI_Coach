import numpy as np
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import pyshine as ps

# import math

from numpy import dot
from numpy.linalg import norm

# Define function for computing cosine-similarity from images
#https://stackoverflow.com/a/43043160/45963
def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))

# Define function for computing cosine-similarity from 2-column arrays
# https://stackoverflow.com/questions/72039174/what-is-the-fastest-way-of-calculate-cosine-similarity-between-rows-of-two-same
def cosine_sim(x, y):
    return (x * y).sum(axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

# define function for making keypoint predictions
def make_predictions(img, interpreter, image_size):
        enhance_contrast = 0
        if enhance_contrast == 1:
            # ----------------------------------------------------------------------
            # enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) 
            # and LAB Color space https://www.xrite.com/blog/lab-color-space
            # ----------------------------------------------------------------------
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

        return keypoints_with_scores

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

# Make predictions:
def make_predictions_compact(input_image, interpreter):
    # SETUP INPUT and OUTPUT
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_scores

# define function for drawing the keypoints

def draw_keypoints(frame, keypoints_with_scores, confidence_threshold):
    y, x, c = frame.shape # (y,x) = coordinates; c = channels
    # convert the normalized coordinates to pixel coordinates:
    # so basically multiply keypoints (e.g. left_eye) with the 
    # input frame dimension (e.g. 480, 640, 1)
    rescaled = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    
    for kp in rescaled:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            # draw circles at a particular x-y position of the frame. In this case circle 
            # size is "4", color RGB (0,255,0), -1 (fills the circle)
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0),-1)
            cv2.circle(frame, (int(kx), int(ky)), 6, (224,255,255), 2)
            
            # conf_sc = float("{:.1}".format(kp_conf))
            #     # print confidence scores on frame
            #     # Using cv2.putText()
            # cv2.putText(
            #     frame,
            #     text = str(conf_sc),
            #     org = (int(kx)+15, int(ky)+15),
            #     fontFace = cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale = 0.9,
            #     color = (224,255,255),
            #     thickness = 1
            #     )

def draw_keypoints_initial(frame, keypoints_with_scores, confidence_threshold):
    y, x, c = frame.shape # (y,x) = coordinates; c = channels
    # convert the normalized coordinates to pixel coordinates:
    # so basically multiply keypoints (e.g. left_eye) with the 
    # input frame dimension (e.g. 480, 640, 1)
    rescaled = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    
    for kp in rescaled:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            # draw circles at a particular x-y position of the frame. In this case circle 
            # size is "4", color RGB (0,255,0), -1 (fills the circle)
            cv2.circle(frame, (int(kx), int(ky)), 6, (55,55,255),-1)
            cv2.circle(frame, (int(kx), int(ky)), 6, (224,255,255), 2)
            
            # conf_sc = float("{:.1}".format(kp_conf))
            #     # print confidence scores on frame
            #     # Using cv2.putText()
            # cv2.putText(
            #     frame,
            #     text = str(conf_sc),
            #     org = (int(kx)+15, int(ky)+15),
            #     fontFace = cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale = 0.9,
            #     color = (224,255,255),
            #     thickness = 1
            #     )

# define function for drawing the line connections

def draw_connections(frame, keypoints_with_scores, edges, confidence_threshold):

    # grab frame coordinates
    y, x, c = frame.shape

    # apply transformation
    rescaled = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = rescaled[p1]
        y2, x2, c2 = rescaled[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

# def function to draw FPS on frame
def draw_FPS(frame, counter, fps, start_time):

    # Visualization parameters
    row_size = 40  # pixels
    left_margin = 34  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 2
    font_thickness = 2
    fps_avg_frame_count = 10

        # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

# function to run TFLite pose classification model:
def classifier(keypoints_with_scores):
    
    # Load label list
    label_path = './pose_classification_results/pose_classifier/pose_labels.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = [line.rstrip() for line in lines]
    
    # Initialize the TFLite model.
    model_path_classifier = "./pose_classification_results/pose_classifier/pose_classifier.tflite"
    interpreter_PoseClass = tf.lite.Interpreter(model_path=model_path_classifier, num_threads=4)
    interpreter_PoseClass.allocate_tensors()

    # define classifier
    input_index = interpreter_PoseClass.get_input_details()[0]['index']
    output_index = interpreter_PoseClass.get_output_details()[0]['index']

    # swap x and y coordinates
    dummy = keypoints_with_scores.copy()
    keypoints_with_scores[:,:,:,0] = dummy[:,:,:,1]
    keypoints_with_scores[:,:,:,1] = dummy[:,:,:,0]

    input_tensor = keypoints_with_scores
    input_tensor = np.array(input_tensor).flatten().astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    # Set the input and run inference.
    interpreter_PoseClass.set_tensor(input_index, input_tensor)
    interpreter_PoseClass.invoke()

    # Extract the output and squeeze the batch dimension
    output = interpreter_PoseClass.get_tensor(output_index)
    output = np.squeeze(output, axis=0)

    # Sort output by descending probability
    prob_descending = sorted(
        range(len(output)), key=lambda k: output[k], reverse=True)
    prob_list_labels = [
        labels[idx] for idx in prob_descending
    ]

    prob_list_scores = [
        output[idx] for idx in prob_descending
    ]
    # the first two outputs are the sorted labels and scores, the last two outputs are for unsorted scores
    return prob_list_labels, prob_list_scores, output, labels


def draw_class_prediction_results(keypoints_with_scores, prob_list_labels, prob_list_scores, frame):
    # Visualization parameters
    keypoint_detection_threshold = 0.1
    classification_results_to_show = 1
    row_size = 30  # pixels
    left_margin = 34  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 2
    font_thickness = 2

    # Check if all keypoints are detected
    min_score = min(keypoints_with_scores[:,:,:,2].flatten())
    if min_score < keypoint_detection_threshold:
        error_text = 'Not enough keypoints detected.'
        text_location = (left_margin, 2 * row_size)
        cv2.putText(frame, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible.'
        text_location = (left_margin, 3 * row_size)
        cv2.putText(frame, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

    else:                

        # Draw the classification results:
        for i in range(classification_results_to_show):
            class_name = prob_list_labels[i]
            probability = round(prob_list_scores[i], 2)
            result_text = class_name + ' (' + str(probability) + ')'
            text_location = (left_margin, (i + 2) * row_size)

            # x,y,w,h = 0,0,600,125
            # # Create background rectangle with color
            # cv2.rectangle(frame, (x,x), (x + w, y + h), (0,0,0), -1)

            # cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            #             font_size, text_color, font_thickness)

            ps.putBText(frame,result_text,text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(1,1,1))

def draw_cosine_similarity(keypoints_with_scores, cos_sim_score_kpt, mse, frame):
    # Visualization parameters
    keypoint_detection_threshold = 0.1
    classification_results_to_show = 1
    row_size = 30  # pixels
    left_margin = 34  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 2
    font_thickness = 2

    # Check if all keypoints are detected
    min_score = min(keypoints_with_scores[:,:,:,2].flatten())
    if min_score < keypoint_detection_threshold:
        error_text = 'Not enough keypoints detected.'
        text_location = (left_margin, 2 * row_size)
        cv2.putText(frame, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible.'
        text_location = (left_margin, 3 * row_size)
        cv2.putText(frame, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

    else:                

        # Draw the classification results:
        for i in range(classification_results_to_show):
            
            # probability = round(cos_sim_score_frame, 2)
            # result_text = 'Cosine_Sim_score_Im' + ' (' + str(probability) + ')'
            # text_location = (left_margin, (1 + 2) * row_size)
            # cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            #             font_size, text_color, font_thickness)
            
            probability2 = round(cos_sim_score_kpt, 2)
            result_text2 = 'Cosine_Sim_Score' + ' (' + str(probability2) + ')'
            probability3 = round(mse, 2)
            result_text3 = 'MSE' + ' (' + str(probability3) + ')'

            text_location2 = (left_margin, (1 + 2) * row_size)
            
            # https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
            # cosine similarity
            ps.putBText(frame,result_text2,text_offset_x=20,text_offset_y=30 + row_size,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(1,1,1))
            # mean square error
            ps.putBText(frame,result_text3,text_offset_x=20,text_offset_y=40 + 2 * row_size,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(1,1,1))
            # cv2.putText(frame, result_text2, text_location2, cv2.FONT_HERSHEY_PLAIN,
            #             font_size, text_color, font_thickness)

 
def getAngle(a, b, c):
    # ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    # return ang + 360 if ang < 0 else ang

    a = a[:2]
    b = b[:2]
    c = c[:2]
       

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    return angle

def draw_angles(frame, keypoints_with_scores, confidence_threshold):
    y, x, c = frame.shape # (y,x) = coordinates; c = channels
    # convert the normalized coordinates to pixel coordinates:
    # so basically multiply keypoints (e.g. left_eye) with the 
    # input frame dimension (e.g. 480, 640, 1)
    rescaled = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))


    #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
    left_arm_and_torso = getAngle(
        keypoints_with_scores[0][0][6],
        keypoints_with_scores[0][0][5],
        keypoints_with_scores[0][0][7],
    )

    #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
    right_arm_and_torso = getAngle(
        keypoints_with_scores[0][0][8],
        keypoints_with_scores[0][0][6],
        keypoints_with_scores[0][0][5]
    )

    #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
    left_arm = getAngle(
        keypoints_with_scores[0][0][5],
        keypoints_with_scores[0][0][7],
        keypoints_with_scores[0][0][9],
    )
    
    #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
    right_arm = getAngle(
        keypoints_with_scores[0][0][10],
        keypoints_with_scores[0][0][8],
        keypoints_with_scores[0][0][6],
    )

    #left hip: 13, 11, 12 (hip, knee and ankle)
    left_hip = getAngle(
        keypoints_with_scores[0][0][13],
        keypoints_with_scores[0][0][11],
        keypoints_with_scores[0][0][12],
    )

    #right hip: 14, 12, 11 (hip, knee and ankle)
    rigth_hip = getAngle(
        keypoints_with_scores[0][0][14],
        keypoints_with_scores[0][0][12],
        keypoints_with_scores[0][0][11],
    )

    #left leg: 11, 13, 15 (hip, knee and ankle)
    left_leg = getAngle(
        keypoints_with_scores[0][0][11],
        keypoints_with_scores[0][0][13],
        keypoints_with_scores[0][0][15],
    )

    #right leg : 12, 14, 16 (hip, knee and ankle)
    right_leg = getAngle(
        keypoints_with_scores[0][0][12],
        keypoints_with_scores[0][0][14],
        keypoints_with_scores[0][0][16],
    )

    # distance between left and right shoulder
    #shoulder_dist = np.linalg.norm(keypoints_with_scores[0][0][5]-keypoints_with_scores[0][0][6])

    # distance between left and right ankle
    #ankle_dist = np.linalg.norm(keypoints_with_scores[0][0][15]-keypoints_with_scores[0][0][16])

    # relative distance between left and right ankle with respect to shoulder distance
    #rel_ankle_dist = ankle_dist/shoulder_dist

    y_scale = 0.95
    font_size = 0.8
    font_thickness = 2
    color = (0,0,255)
    # color = (255,0,0)

    
    i=-1
    for kp in rescaled:
        i+=1
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            if i == 5:
            # draw angles at a particular x-y position of the frame:            
                left_arm_and_torso = int(left_arm_and_torso)
                    # print confidence scores on frame
                    # Using cv2.putText()
                cv2.putText(
                    frame,
                    text = str(left_arm_and_torso),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )

            if i == 6:
            # draw angles at a particular x-y position of the frame:            
                right_arm_and_torso = int(right_arm_and_torso)
                    # print confidence scores on frame
                    # Using cv2.putText()
                cv2.putText(
                    frame,
                    text = str(right_arm_and_torso),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )

            if i == 7:
            # draw angles at a particular x-y position of the frame:            
                left_arm = int(left_arm)
                    # print confidence scores on frame
                    # Using cv2.putText()
                cv2.putText(
                    frame,
                    text = str(left_arm),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )


            if i == 8:
            # draw angles at a particular x-y position of the frame:            
                    right_arm = int(right_arm)
                    # print confidence scores on frame
                    # Using cv2.putText()
                    cv2.putText(
                    frame,
                    text = str(right_arm),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )

            if i == 11:
            # draw angles at a particular x-y position of the frame:            
                left_hip = int(left_hip)
                    # print confidence scores on frame
                    # Using cv2.putText()
                cv2.putText(
                    frame,
                    text = str(left_hip),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )


            if i == 12:
            # draw angles at a particular x-y position of the frame:            
                    rigth_hip = int(rigth_hip)
                    # print confidence scores on frame
                    # Using cv2.putText()
                    cv2.putText(
                    frame,
                    text = str(rigth_hip),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )

            if i == 13:
            # draw angles at a particular x-y position of the frame:            
                left_leg = int(left_leg)
                    # print confidence scores on frame
                    # Using cv2.putText()
                cv2.putText(
                    frame,
                    text = str(left_leg),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )


            if i == 14:
            # draw angles at a particular x-y position of the frame:            
                    right_leg = int(right_leg)
                    # print confidence scores on frame
                    # Using cv2.putText()
                    cv2.putText(
                    frame,
                    text = str(right_leg),
                    org = (int(kx), int(ky*y_scale)),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = font_size,
                    color = color,
                    thickness = font_thickness
                    )

            # if i == 16:
            # # draw angles at a particular x-y position of the frame:            
            #         rel_ankle_dist = float("{:.2}".format(rel_ankle_dist))
            #         # print confidence scores on frame
            #         # Using cv2.putText()
            #         cv2.putText(
            #         frame,
            #         text = 'rel_ankle_dist: ' + str(rel_ankle_dist),
            #         org = (int(kx), int(ky*y_scale)),
            #         fontFace = cv2.FONT_HERSHEY_DUPLEX,
            #         fontScale = font_size,
            #         color = color,
            #         thickness = font_thickness
            #         )


