import numpy as np

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

def reference_pose_angles(keypoints_with_scores):

    #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
    left_arm_and_torso = getAngle(
        keypoints_with_scores[6],
        keypoints_with_scores[5],
        keypoints_with_scores[7]
    )

    #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
    right_arm_and_torso = getAngle(
        keypoints_with_scores[8],
        keypoints_with_scores[6],
        keypoints_with_scores[5]
    )

    #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
    left_arm = getAngle(
        keypoints_with_scores[5],
        keypoints_with_scores[7],
        keypoints_with_scores[9],
    )
    
    #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
    right_arm = getAngle(
        keypoints_with_scores[10],
        keypoints_with_scores[8],
        keypoints_with_scores[6]
    )

    #left hip: 13, 11, 12 (hip, knee and ankle)
    left_hip = getAngle(
        keypoints_with_scores[13],
        keypoints_with_scores[11],
        keypoints_with_scores[12]
    )

    #right hip: 14, 12, 11 (hip, knee and ankle)
    rigth_hip = getAngle(
        keypoints_with_scores[14],
        keypoints_with_scores[12],
        keypoints_with_scores[11]
    )

    #left leg: 11, 13, 15 (hip, knee and ankle)
    left_leg = getAngle(
        keypoints_with_scores[11],
        keypoints_with_scores[13],
        keypoints_with_scores[15]
    )

    #right leg : 12, 14, 16 (hip, knee and ankle)
    right_leg = getAngle(
        keypoints_with_scores[12],
        keypoints_with_scores[14],
        keypoints_with_scores[16]
    )

    # EXTRA angle calculations:
#-------------------------

    #left leg and torso: 13, 11, 5 (knee, hip and shoulder)
    left_leg_torso = getAngle(
        keypoints_with_scores[13],
        keypoints_with_scores[11],
        keypoints_with_scores[5]
    )

    #right leg and torso: 14, 12, 6 (knee, hip and shoulder)
    right_leg_torso = getAngle(
        keypoints_with_scores[14],
        keypoints_with_scores[12],
        keypoints_with_scores[6]
    )

    #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
    left_arm_torso = getAngle(
        keypoints_with_scores[7],
        keypoints_with_scores[5],
        keypoints_with_scores[11]
    )

    #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
    right_arm_torso = getAngle(
        keypoints_with_scores[8],
        keypoints_with_scores[6],
        keypoints_with_scores[12]
    )
    
    #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)
    right_arm_left_arm = getAngle(
        keypoints_with_scores[10],
        keypoints_with_scores[6],
        keypoints_with_scores[9]
    )

    return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_hip, rigth_hip, left_leg, right_leg, \
              left_leg_torso, right_leg_torso, left_arm_torso, right_arm_torso, right_arm_left_arm

def pose_angles(keypoints_with_scores):

    #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
    left_arm_and_torso = getAngle(
        keypoints_with_scores[0][0][6],
        keypoints_with_scores[0][0][5],
        keypoints_with_scores[0][0][7]
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
        keypoints_with_scores[0][0][9]
    )
    
    #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
    right_arm = getAngle(
        keypoints_with_scores[0][0][10],
        keypoints_with_scores[0][0][8],
        keypoints_with_scores[0][0][6]
    )

    #left hip: 13, 11, 12 (hip, knee and ankle)
    left_hip = getAngle(
        keypoints_with_scores[0][0][13],
        keypoints_with_scores[0][0][11],
        keypoints_with_scores[0][0][12]
    )

    #right hip: 14, 12, 11 (hip, knee and ankle)
    rigth_hip = getAngle(
        keypoints_with_scores[0][0][14],
        keypoints_with_scores[0][0][12],
        keypoints_with_scores[0][0][11]
    )

    #left leg: 11, 13, 15 (hip, knee and ankle)
    left_leg = getAngle(
        keypoints_with_scores[0][0][11],
        keypoints_with_scores[0][0][13],
        keypoints_with_scores[0][0][15]
    )

    #right leg : 12, 14, 16 (hip, knee and ankle)
    right_leg = getAngle(
        keypoints_with_scores[0][0][12],
        keypoints_with_scores[0][0][14],
        keypoints_with_scores[0][0][16]
    )
# EXTRA angle calculations:
#-------------------------

    #left leg and torso: 13, 11, 5 (knee, hip and shoulder)
    left_leg_torso = getAngle(
        keypoints_with_scores[0][0][13],
        keypoints_with_scores[0][0][11],
        keypoints_with_scores[0][0][5]
    )

    #right leg and torso: 14, 12, 6 (knee, hip and shoulder)
    right_leg_torso = getAngle(
        keypoints_with_scores[0][0][14],
        keypoints_with_scores[0][0][12],
        keypoints_with_scores[0][0][6]
    )

    #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
    left_arm_torso = getAngle(
        keypoints_with_scores[0][0][7],
        keypoints_with_scores[0][0][5],
        keypoints_with_scores[0][0][11]
    )

    #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
    right_arm_torso = getAngle(
        keypoints_with_scores[0][0][8],
        keypoints_with_scores[0][0][6],
        keypoints_with_scores[0][0][12]
    )
    
    #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)
    right_arm_left_arm = getAngle(
        keypoints_with_scores[0][0][10],
        keypoints_with_scores[0][0][6],
        keypoints_with_scores[0][0][9]
    )

    return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_hip, rigth_hip, left_leg, right_leg, \
        left_leg_torso, right_leg_torso, left_arm_torso, right_arm_torso, right_arm_left_arm

# --------------------------------------------------------------------------------------------------------
# The follow angle calculations are specific for each Asana and used in conjunction with cosine similarity
# --------------------------------------------------------------------------------------------------------

def asana_pose_angles_from_frame(keypoints_with_scores, pose_idx):

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

                #left leg and torso: 13, 11, 5 (knee, hip and shoulder)
        left_leg_torso = getAngle(
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][5]
        )

        #right leg and torso: 14, 12, 6 (knee, hip and shoulder)
        right_leg_torso = getAngle(
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][6]
        )

        return left_leg, right_leg, left_arm, right_arm, left_leg_torso, right_leg_torso

# (1) Tadhasana:
# -----------
    elif pose_idx == 1:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7],
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

            #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
        left_arm_torso = getAngle(
            keypoints_with_scores[0][0][7],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][11]
        )

        #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
        right_arm_torso = getAngle(
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][12]
        )

        return left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

# (2) Utthita_Parsvakonasana / Trikonasana:
# ----------------------------------------

    elif pose_idx == 2:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7],
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)
        right_arm_left_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][9]
        )

        return left_leg, right_leg, left_arm, right_arm, right_arm_left_arm

# (3) Warrior_I:
# ------------
    elif pose_idx == 3:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7],
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

         #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
        left_arm_torso = getAngle(
            keypoints_with_scores[0][0][7],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][11]
        )

        #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
        right_arm_torso = getAngle(
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][12]
        )

        return left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

# (4) Warrior_II:
# ------------
    elif pose_idx == 4:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (5) Utthita_Parsvakonasana:
# ------------
    elif pose_idx == 5:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (6) Halasana:
# ------------
    elif pose_idx == 6:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (7) Chaturanga:
# ------------
    elif pose_idx == 7:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (8) Paschimottanasana:
# ------------
    elif pose_idx == 8:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (9) Prasarita_Padottanasana:
# ------------
    elif pose_idx == 9:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[0][0][6],
            keypoints_with_scores[0][0][5],
            keypoints_with_scores[0][0][7]
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
            keypoints_with_scores[0][0][9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[0][0][10],
            keypoints_with_scores[0][0][8],
            keypoints_with_scores[0][0][6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[0][0][11],
            keypoints_with_scores[0][0][13],
            keypoints_with_scores[0][0][15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[0][0][12],
            keypoints_with_scores[0][0][14],
            keypoints_with_scores[0][0][16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

def asana_pose_angles_from_reference(keypoints_with_scores, pose_idx):

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
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15],
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16],
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9],
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6],
        )

                #left leg and torso: 13, 11, 5 (knee, hip and shoulder)
        left_leg_torso = getAngle(
            keypoints_with_scores[13],
            keypoints_with_scores[11],
            keypoints_with_scores[5]
        )

        #right leg and torso: 14, 12, 6 (knee, hip and shoulder)
        right_leg_torso = getAngle(
            keypoints_with_scores[14],
            keypoints_with_scores[12],
            keypoints_with_scores[6]
        )

        return left_leg, right_leg, left_arm, right_arm, left_leg_torso, right_leg_torso

# (1) Tadhasana:
# -----------
    elif pose_idx == 1:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
        left_arm_torso = getAngle(
            keypoints_with_scores[7],
            keypoints_with_scores[5],
            keypoints_with_scores[11]
        )

        #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
        right_arm_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[12]
        )

        return left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

# (2) Utthita_Parsvakonasana / Trikonasana:
# ----------------------------------------

    elif pose_idx == 2:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #right arm and left arm: 10, 6, 9 (wrist, shoulder and wrist)
        right_arm_left_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[6],
            keypoints_with_scores[9]
        )

        return left_leg, right_leg, left_arm, right_arm, right_arm_left_arm

# (3) Warrior_I:
# ------------
    elif pose_idx == 3:

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

         #left arm and torso: 7, 5, 11 (elbow, shoulder and hip)
        left_arm_torso = getAngle(
            keypoints_with_scores[7],
            keypoints_with_scores[5],
            keypoints_with_scores[11]
        )

        #right arm and torso: 8, 6, 12 (elbow, shoulder and hip)
        right_arm_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[12]
        )

        return left_leg, right_leg, left_arm, right_arm, left_arm_torso, right_arm_torso

# (4) Warrior_II:
# ------------
    elif pose_idx == 4:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (5) Utthita_Parsvakonasana:
# ------------
    elif pose_idx == 5:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (6) Halasana:
# ------------
    elif pose_idx == 6:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (7) Chaturanga:
# ------------
    elif pose_idx == 7:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (8) Paschimottanasana:
# ------------
    elif pose_idx == 8:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg

# (9) Prasarita_Padottanasana:
# ------------
    elif pose_idx == 9:

        #left_arm_and_torso: 6,5,7 (right shoulder, left shoulder and left elbow)
        left_arm_and_torso = getAngle(
            keypoints_with_scores[6],
            keypoints_with_scores[5],
            keypoints_with_scores[7]
        )

        #right_arm_and_torso: 8,6,5 (left elbow, right shoulder and left shoulder)
        right_arm_and_torso = getAngle(
            keypoints_with_scores[8],
            keypoints_with_scores[6],
            keypoints_with_scores[5]
        )

        #left_arm: 5, 7, 9 (shoulder, elbow and wrist)
        left_arm = getAngle(
            keypoints_with_scores[5],
            keypoints_with_scores[7],
            keypoints_with_scores[9]
        )
        
        #right_arm: 10, 8, 6 (shoulder, elbow and wrist)
        right_arm = getAngle(
            keypoints_with_scores[10],
            keypoints_with_scores[8],
            keypoints_with_scores[6]
        )

        #left leg: 11, 13, 15 (hip, knee and ankle)
        left_leg = getAngle(
            keypoints_with_scores[11],
            keypoints_with_scores[13],
            keypoints_with_scores[15]
        )

        #right leg : 12, 14, 16 (hip, knee and ankle)
        right_leg = getAngle(
            keypoints_with_scores[12],
            keypoints_with_scores[14],
            keypoints_with_scores[16]
        )   

        return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_leg, right_leg