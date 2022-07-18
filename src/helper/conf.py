
# Model location:
model_path_thunder_3="./models/lite-model_movenet_singlepose_thunder_3.tflite"
model_path_lightning_3="./models/lite-model_movenet_singlepose_lightning_3.tflite"
model_path_classifier = "./pose_classification_results/pose_classifier/pose_classifier.tflite"

# Classifier label locations
label_path = './pose_classification/pose_classifier/pose_labels.txt'

# Dictionary to map joints of body part
KEYPOINT_DICT = {
    'nose':0,
    'left_eye':1,
    'right_eye':2,
    'left_ear':3,
    'right_ear':4,
    'left_shoulder':5,
    'right_shoulder':6,
    'left_elbow':7,
    'right_elbow':8,
    'left_wrist':9,
    'right_wrist':10,
    'left_hip':11,
    'right_hip':12,
    'left_knee':13,
    'right_knee':14,
    'left_ankle':15,
    'right_ankle':16
}

# define edges for making the line connections;
# map lines to matplotlib color name

EDGES = {
    (0, 1): (255,0,255),
    (0, 2): (255,0,0),
    (1, 3): (255,0,255),
    (2, 4): (255,0,0),
    (0, 5): (255,0,255),
    (0, 6): (255,0,0),
    (5, 7): (255,0,255),
    (7, 9): (255,0,255),
    (6, 8): (255, 153, 255),
    (8, 10): (255, 153, 255),
    (5, 6): (255,255,0),
    (5, 11): (255,255,0),
    (6, 12): (255,255,0),
    (11, 12): (255,255,0),
    (11, 13): (255, 128, 0),
    (13, 15): (255, 128, 0),
    (12, 14): (255,0,255),
    (14, 16): (255,0,255),
}
