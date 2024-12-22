import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load an official model
names = model.names

print(names)
names[0] = "boy"
print(names)
# Predict with the model
# results = model('/home/sean/Omost/74a6963bb8e94bafaf7f12111b7e22e4_0.png',save=True)  # predict on an image

# for r in results:
#     for c in r.boxes.cls:
#         print(names[int(c)])



def is_match_yolo(yolo_model, photo_path, name_list, corresponding_num_list):
    """
    Check if the names match the corresponding number of occurrences in a YOLO model's prediction.

    Args:
        yolo_model (YOLO): The YOLO model instance for detection.
        photo_path (str): Path to the image for prediction.
        name_list (list): List of class names to match.
        corresponding_num_list (list): List of expected counts for each class name.

    Returns:
        bool: True if the counts match, False otherwise.
    """
    if not isinstance(name_list, list) or not isinstance(corresponding_num_list, list):
        raise TypeError("name_list and corresponding_num_list must both be lists.")
    
    if len(name_list) != len(corresponding_num_list):
        raise ValueError("The length of name_list and corresponding_num_list must be the same.")
    
    # Run the YOLO model on the provided image
    try:
        results = yolo_model(photo_path)  # predict on an image
    except Exception as e:
        raise RuntimeError(f"Error during YOLO prediction: {e}")
    
    # Initialize a dictionary to count occurrences of the target names
    detected_count = {name: 0 for name in name_list}
    names = yolo_model.names

    # Count detections
    for r in results:
        for c in r.boxes.cls:
            class_name = names[int(c)]
            if class_name in detected_count:
                detected_count[class_name] += 1

    # Compare the detected counts with the expected counts
    for i, name in enumerate(name_list):
        if detected_count.get(name, 0) != corresponding_num_list[i]:
            return False

    return True
