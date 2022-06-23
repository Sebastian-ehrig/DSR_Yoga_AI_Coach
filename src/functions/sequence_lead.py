from ast import arg
import pygame
import numpy as np
import time

from functions.helper import *
from functions.pose_calc import *
from functions.corrections import *

# https://stackoverflow.com/questions/70223324/play-a-sound-asynchronously-in-a-while-loop

pygame.init()
MUSIC_ENDED = pygame.USEREVENT

def playSound(filename):
    pygame.mixer.quit() 
    pygame.mixer.init(44100, -16, 2, 4096)
    pygame.mixer.music.set_endevent(MUSIC_ENDED)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()


    # while pygame.mixer.music.get_busy() == True:
    #     continue
 
# define the countdown func.
def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
        return timer

def yoga_sequence_lead(keypoints_reference_pose, keypoints_with_scores, pose_idx, seq_step, mse):

    # get angles of closest matching pose
    pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
    pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

    # calculate angle differences between reference and current frame
    # pose_angle_differences = pose_angles_reference_img - pose_angles_current_frame
    pose_angle_differences_abs = abs(pose_angles_reference_img - pose_angles_current_frame)

    # get index of largest angular difference
    # maxDiff_idx = np.where(pose_angle_differences == np.amax(pose_angle_differences))[0][0]
    
    angl_thresh_tight = 10
    mse_thresh = 100
    time_in = time.time()

# Voice Commands for sequence lead:
# ---------------------------------

    Intro = "./src/functions/intro_audio/mixkit-simple-game-countdown-921.ogg"
    Inhale_Exhale = "./src/functions/sequence_commands/Inhale_Exhale.ogg"
    # Transition 1
    Tadhasana_to_Warrior1 = "./src/functions/sequence_commands/Tadhasana_to_Warrior1.ogg"
    Inhale_Tadhasana_to_Warrior1 = "./src/functions/sequence_commands/Inhale_Tadhasana_to_Warrior1.ogg"
    # Transition 2
    Warrior1_to_Warrior2 = "./src/functions/sequence_commands/Warrior1_to_Warrior2.ogg"
    Inhale_Warrior1_to_Warrior2 = "./src/functions/sequence_commands/Inhale_Warrior1_to_Warrior2.ogg"
    # Transition 3
    Warrior2_to_Trikonasana = "./src/functions/sequence_commands/Warrior2_to_Trikonasana.ogg"
    Inhale_Warrior2_to_Trikonasana = "./src/functions/sequence_commands/Inhale_Warrior2_to_Trikonasana.ogg"
    # Transition 4
    Trikonasana_to_Tadhasana = './src/functions/sequence_commands/Trikonasana_to_Tadhasana.ogg'
    Inhale_Trikonasana_to_Tadhasana = './src/functions/sequence_commands/Inhale_Trikonasana_to_Tadhasana.ogg'

  
# Selected Asanas:
# ----------------
# 0: Downward_Facing_Dog
# 1: Tadhasana
# 2: Utthita_Parsvakonasana
# 3: Warrior_I
# 4: Warrior_II

# ---------------------------------
# Start of sequence:
# ---------------------------------
    # playSound(Intro)

    if seq_step == 0:

        # Tadhasana:
        # -----------

            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
            #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso
            # make pose-correction suggestions

        if (pose_idx == 1 and # Tadhasana
            # pose_angle_differences_abs[0] < angl_thresh_tight and #left_leg
            # pose_angle_differences_abs[1] < angl_thresh_tight and #right_leg
            # pose_angle_differences_abs[4] < angl_thresh_tight and #left_arm_torso
            # pose_angle_differences_abs[5] < angl_thresh_tight and #right_arm_torso
            mse < 50
            ):
            time_in = time.time()
            playSound(Inhale_Tadhasana_to_Warrior1)

            # for event in pygame.event.get():
            #     if event.type == MUSIC_ENDED:
            #         print('music end event')
                    
            # Transition 1
            # ------------
            # playSound(Tadhasana_to_Warrior1)
            # print('music end event')
            seq_step = seq_step + 1

    if seq_step == 1:
        # Warrior 1:
        # -----------

            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
            #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)

            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

        if (pose_idx == 3 and # Warrior_I
            # pose_angle_differences_abs[1] > angl_thresh_tight and #right_leg
            # pose_angle_differences_abs[5] > angl_thresh_tight and #right_arm_torso
            mse < mse_thresh
            ):
            time_in = time.time()
            # playSound(Inhale_Exhale)

            # Transition 2
            # -----------
            playSound(Inhale_Warrior1_to_Warrior2)

            seq_step = seq_step + 1

    if seq_step == 2:
        # Warrior 2:
        # -----------
                #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
                #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
                #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
                #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
                #left leg: 11, 13, 15 (hip, knee and ankle)
                #right leg : 12, 14, 16 (hip, knee and ankle)

                # Array_Order: left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

        if(pose_idx == 4 and # Warrior_II
            # pose_angle_differences_abs[0] > angl_thresh_tight and 
            # pose_angle_differences_abs[2] > angl_thresh_tight and
            # pose_angle_differences_abs[1] > angl_thresh_tight and 
            # pose_angle_differences_abs[3] > angl_thresh_tight and
            mse < mse_thresh
            ):
            time_in = time.time()
            # startTime = time.time()
            # playSound(Inhale_Exhale)

            # Transition 3
            # -----------
            playSound(Inhale_Warrior2_to_Trikonasana)

            seq_step = seq_step + 1

    if seq_step == 3:
        # Trikonasana:
        # -----------

                #left leg: 11, 13, 15 (hip, knee and ankle)
                #right leg : 12, 14, 16 (hip, knee and ankle)
                #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
                #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
                #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)

                # Array_Order: left_leg, right_leg, left_arm, right_arm, right_arm_left_arm

        if (pose_idx == 2 and # Utthita_Parsvakonasana/Trikonasana
            # pose_angle_differences_abs[1] > angl_thresh_tight and 
            # pose_angle_differences_abs[4] > angl_thresh_tight and 
            # pose_angle_differences_abs[2] > angl_thresh_tight and 
            # pose_angle_differences_abs[3] > angl_thresh_tight and
            mse < mse_thresh
            ):
            time_in = time.time()
            # startTime = time.time()
            # playSound(Inhale_Exhale)

            # Transition 4
            # -----------
            playSound(Inhale_Trikonasana_to_Tadhasana)

            seq_step = seq_step + 1

    if seq_step == 4:
        # Tadhasana:
        # -----------
            #left leg: 11, 13, 15 (hip, knee and ankle)
            #right leg : 12, 14, 16 (hip, knee and ankle)
            #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
            #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
            #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
            # Array_Order: left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

        if (pose_idx == 1 and # Tadhasana
            # pose_angle_differences_abs[0] < angl_thresh_tight and #left_leg
            # pose_angle_differences_abs[1] < angl_thresh_tight and #right_leg
            # pose_angle_differences_abs[4] < angl_thresh_tight and #left_arm_torso
            # pose_angle_differences_abs[5] < angl_thresh_tight and #right_arm_torso
            mse < 50
            ):
            time_in = time.time()
            # startTime = time.time()
            playSound(Inhale_Exhale)

            seq_step = seq_step + 1

    return seq_step, time_in