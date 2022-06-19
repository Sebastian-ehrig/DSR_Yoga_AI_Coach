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
        keypoints_with_scores[7],
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
        keypoints_with_scores[6],
    )

    #left hip: 13, 11, 12 (hip, knee and ankle)
    left_hip = getAngle(
        keypoints_with_scores[13],
        keypoints_with_scores[11],
        keypoints_with_scores[12],
    )

    #right hip: 14, 12, 11 (hip, knee and ankle)
    rigth_hip = getAngle(
        keypoints_with_scores[14],
        keypoints_with_scores[12],
        keypoints_with_scores[11],
    )

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

    return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_hip, rigth_hip, left_leg, right_leg

def pose_angles(keypoints_with_scores):

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

    return left_arm_and_torso, right_arm_and_torso, left_arm, right_arm, left_hip, rigth_hip, left_leg, right_leg
