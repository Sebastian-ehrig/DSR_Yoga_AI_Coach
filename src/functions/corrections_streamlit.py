import numpy as np
import pygame
import streamlit as st

# pygame.mixer.pre_init(44100, -16, 2, 4096) #frequency, size, channels, buffersize
# pygame.init() #turn all of pygame on.
# pygame.mixer.init()

from functions.helper import *
from functions.pose_calc import *

def playSound_ST(filename):
        audio_file = open(filename, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/ogg')


# def playSound_ST(filename):
#         sound = st.empty()
#         sound.markdown(filename, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
#         time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
#         sound.empty()  # optionally delete the element afterwards


def correct_angles_ST(keypoints_reference_pose, keypoints_with_scores, pose_idx):

    keypoint_detection_threshold = 0.1

# Check if all keypoints are detected
    min_score = min(keypoints_with_scores[:,:,:,2].flatten())
    if min_score < keypoint_detection_threshold:
        pass #print("Not enough keypoints detected")
    else:                

        # get angles of closest matching pose
        pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
        pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

        # calculate angle differences between reference and current frame
        # pose_angle_differences = pose_angles_reference_img - pose_angles_current_frame
        pose_angle_differences_abs = abs(pose_angles_reference_img - pose_angles_current_frame)

        # get index of largest angular difference
        # maxDiff_idx = np.where(pose_angle_differences == np.amax(pose_angle_differences))[0][0]
        
        angl_thresh = 20

    # Selected Asanas:
    # ----------------
    # Downward_Facing_Dog
    # Tadhasana
    # Utthita_Parsvakonasana
    # Warrior_I
    # Warrior_II

    # Voice Commands for pose corrections:
    # ------------------------------------
        Lift_the_backarm_up = "./src/functions/voice_commands/Lift_the_backarm_up.ogg"
        Bend_the_knee = "./src/functions/voice_commands/Bend_the_knee.ogg"
        Lengthen_the_spine = './src/functions/voice_commands/Lengthen_the_spine.ogg'
        Straighten_front_leg = "./src/functions/voice_commands/Straighten_front_leg.ogg"
        Keep_arms_in_one_line = "./src/functions/voice_commands/Keep_arms_in_one_line.ogg"
        Lift_the_arms_higher = "./src/functions/voice_commands/Lift_the_arms_higher.ogg"


    # Voice command paths for html code:
    # ----------------------------------

        # Lift_the_backarm_up = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Lift_the_backarm_up.ogg" type="audio/ogg">
        #         </audio>
        #         """

        # Bend_the_knee = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Bend_the_knee.ogg" type="audio/ogg">
        #         </audio>
        #         """

        # Lengthen_the_spine = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Lengthen_the_spine.ogg" type="audio/ogg">
        #         </audio>
        #         """

        # Straighten_front_leg = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Straighten_front_leg.ogg" type="audio/ogg">
        #         </audio>
        #         """

        # Keep_arms_in_one_line = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Keep_arms_in_one_line.ogg" type="audio/ogg">
        #         </audio>
        #         """

        # Lift_the_arms_higher = """
        #         <audio controls autoplay>
        #         <source src="./src/functions/voice_commands/Lift_the_arms_higher.ogg" type="audio/ogg">
        #         </audio>
        #         """


    # (0) Downward_Facing_Dog:
    # --------------------

        if pose_idx == 0:

            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left leg and torso: 13, 11, 5 (knee, hip and shoulder)
            #right leg and torso: 14, 12, 6 (knee, hip and shoulder)
            
            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_leg_torso, right_leg_torso

            if pose_angle_differences_abs[4] > angl_thresh or pose_angle_differences_abs[5] > angl_thresh:
                playSound_ST(Straighten_front_leg)

    # (1) Tadhasana:
    # -----------
        elif pose_idx == 1:
            
            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
            #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

            if (pose_angle_differences_abs[0] > angl_thresh or 
                pose_angle_differences_abs[1] > angl_thresh
                ):
                playSound_ST(Lengthen_the_spine)

    # (2) Trikonasana:
    # ----------------------------------------

        elif pose_idx == 2:

            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)

            # Array_Order: left_leg, right_leg, left_arm, right_arm, right_arm_left_arm

            if pose_angle_differences_abs[1] > angl_thresh:
                playSound_ST(Straighten_front_leg)

            elif (pose_angle_differences_abs[4] > angl_thresh or 
                pose_angle_differences_abs[2] > angl_thresh or 
                pose_angle_differences_abs[3] > angl_thresh
                ):
                playSound_ST(Keep_arms_in_one_line)

    # (3) Warrior_I:
    # ------------
        elif pose_idx == 3:

            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
            #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)

            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso
        
            if pose_angle_differences_abs[1] > angl_thresh:
                playSound_ST(Bend_the_knee)

            elif pose_angle_differences_abs[5] > angl_thresh:
                playSound_ST(Lift_the_arms_higher)

    # (4) Warrior_II:
    # ------------
        elif pose_idx == 4:

            #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
            #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)

            # Array_Order: left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

            if (pose_angle_differences_abs[0] > 30 and 
                pose_angle_differences_abs[2] > 30 or 
                pose_angle_differences_abs[1] > 30 and 
                pose_angle_differences_abs[3] > 30
                ):
                playSound_ST(Lift_the_backarm_up)

            elif pose_angle_differences_abs[5]  > 30:
                playSound_ST(Bend_the_knee)
