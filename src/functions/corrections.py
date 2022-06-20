from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import ffpyplayer

from functions.helper import *
from functions.pose_calc import *

def correct_angles(keypoints_reference_pose, keypoints_with_scores, pose_idx):

    # get angles of closest matching pose
    pose_angles_reference_img = np.array(asana_pose_angles_from_reference(keypoints_reference_pose, pose_idx))
    pose_angles_current_frame = np.array(asana_pose_angles_from_frame(keypoints_with_scores, pose_idx))

    # calculate angle differences between reference and current frame
    pose_angle_differences = pose_angles_reference_img - pose_angles_current_frame
    pose_angle_differences_abs = abs(pose_angles_reference_img - pose_angles_current_frame)

    # get index of largest angular difference
    maxDiff_idx = np.where(pose_angle_differences == np.amax(pose_angle_differences))[0][0]

# Selected Asanas:
# ----------------
# Downward_Facing_Dog
# Tadhasana
# Utthita_Parsvakonasana
# Warrior_I
# Warrior_II

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

        # ankle, hip, right_shoulder and Head in one line
        song = AudioSegment.from_wav("./src/functions/voice_commands/Lengthen_the_spine.wav")

        if pose_angle_differences_abs[0] > 20 or pose_angle_differences_abs[1] > 20:
            play(song)

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

        # ankle, hip, right_shoulder and Head in one line
        song = AudioSegment.from_wav("./src/functions/voice_commands/Lengthen_the_spine.wav")

        if pose_angle_differences_abs[0] > 20 or pose_angle_differences_abs[1] > 20:
            play(song)

# (2) Utthita_Parsvakonasana / Trikonasana:
# ----------------------------------------

    elif pose_idx == 2:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        #right leg : 12, 14, 16 (hip, knee and ankle)
        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)

        # Array_Order: left_leg, right_leg, left_arm, right_arm, right_arm_left_arm

        song_3 = AudioSegment.from_wav("./src/functions/voice_commands/Straighten_front_leg.wav")
        song_4 = AudioSegment.from_wav("./src/functions/voice_commands/Keep_arms_in_one_line.wav")

        if pose_angle_differences_abs[1] > 20:
            play(song_3)

        if pose_angle_differences_abs[4] > 20 or pose_angle_differences_abs[2] > 20 or pose_angle_differences_abs[3] > 20:
            play(song_4)

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

        song = AudioSegment.from_wav("./src/functions/voice_commands/Bend_the_knee.wav")
        song_1 = AudioSegment.from_wav("./src/functions/voice_commands/Lift_the_arms_higher")
    
        if pose_angle_differences_abs[1] > 20:
            play(song)

        if pose_angle_differences_abs[5] > 20:
            play(song_1)


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

        song_2 = AudioSegment.from_wav("./src/functions/voice_commands/Bend_the_knee.wav")
        song = AudioSegment.from_wav("./src/functions/voice_commands/Bend_the_knee.wav")

        if pose_angle_differences_abs[0] > 20 and pose_angle_differences_abs[2] > 20 or pose_angle_differences_abs[1] > 20 and pose_angle_differences_abs[3] > 20:
            play(song)

        if pose_angle_differences_abs[5]  > 20:
            play(song_2)