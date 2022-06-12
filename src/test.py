# Load label list
label_path = './pose_classification_results/pose_classifier/pose_labels.txt'
with open(label_path, 'r') as f:
    lines = f.readlines()
    labels = [line.rstrip() for line in lines]
1