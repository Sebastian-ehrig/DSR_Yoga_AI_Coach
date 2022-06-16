from pydub import AudioSegment
from pydub.playback import play
from numpy import np

# import the time module
import time
  
# define the countdown func.
def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    print('Fire in the hole!!')
  
  
# input time in seconds
t = input("5: ")
  
# function call
countdown(int(t))

from functions.helper import getAngle

def correct_angles(frame, keypoints_with_scores, confidence_threshold):
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


# Pose corrections 

# Tadhasana

# ankle, hip, right_shoulder and Head in one line
    song = AudioSegment.from_wav("./voice_commands/Lengthen_the_spine.wav")

    if right_leg > 180 or right_leg < 180:  
        play(song)

# Start of sequence:
    song_a = AudioSegment.from_wav("./voice_commands/Tadhasana.wav")

    if right_leg and right_arm == 180:  
        play(song_a)
    
    # input time in seconds
    t = input("2: ")
  
    # function call
    countdown(int(t))

# Transition 1
    song_b = AudioSegment.from_wav("./voice_commands/Tadhasana_to_warrior1.wav")

    if right_leg and right_arm_and_torso == 90:  
        play(song_b)

# Warrior 1
    song = AudioSegment.from_wav("./voice_commands/Bend_the_knee.wav")
    song_1 = AudioSegment.from_wav("./voice_commands/Lift_the_arms_higher")
    
    if right_leg  > 90:
        play(song)


    if right_arm_and_torso < 120:
        play(song_1)

# Transition 2
    song_b = AudioSegment.from_wav("./voice_commands/Warrior1_Warrior2.wav")

    if right_leg and right_arm == 180:  
        play(song_a)

# Warrior 2
    song_2 = AudioSegment.from_wav("./voice_commands/Lift_the_backarm_up.wav")
    song = AudioSegment.from_wav("./voice_commands/Bend_the_knee.wav")
    
    if left_arm_and_torso and right_arm > 180 or left_arm_and_torso and right_arm < 180:
        play(song)
    
    if right_leg  > 90:
        play(song_2)


# Trikonasana
    song_3 = AudioSegment.from_wav("./voice_commands/straighten_front_leg.wav")
    song_4 = AudioSegment.from_wav("./voice_commands/Keep_arms_in_one_line.wav")

    if right_leg < 180:
        play(song_3)

    if left_arm_and_torso and right_arm > 180 or left_arm_and_torso and right_arm < 180:
        play(song_4)

# Transition 3
    song_c = AudioSegment.from_wav("./voice_commands/Warrior2_Trikonasana.wav")

    if right_leg and right_arm == 180:  
        play(song_c)
