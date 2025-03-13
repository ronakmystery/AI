import cv2
import numpy as np
from cnn import classify_single_image  # Import function from Bird Classifier
import matplotlib.pyplot as plt

def classify(model, image_path):
    """Detects objects, extracts bounding boxes, and re-classifies each cropped region using the Bird Model."""
    
    # Load full image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]  # Get image dimensions

    
    # Run YOLO on the full image
    results = model(image)

    detected_objects = []
    
    for r in results:
        class_ids = r.boxes.cls.cpu().numpy().astype(int)  # YOLO Class IDs
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)

        for i, (class_id, box) in enumerate(zip(class_ids, boxes)):
            x1, y1, x2, y2 = map(int, box)  # Convert bbox to integers
            cropped_object = image[y1:y2, x1:x2]  # Extract detected object

            # Compute bounding box center
            x_center = (x1 + x2) / 2  
            # Determine if object is on the left or right side
            location = "left" if x_center < (img_width / 2) else "right"

            # Convert OpenCV image (BGR) to PIL format for Bird Model

            # Classify using Bird Model
            bird_class_name = classify_single_image(cropped_object)  # Get new label

            # Append detected objects with updated label
            detected_objects.append({
                "class": bird_class_name,  # Use Bird Model class
                "bbox": [float(coord) for coord in box],  # Bounding box coordinates
                "location":location
            })

            # Draw bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(image, bird_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Save updated annotated image
    cv2.imwrite('bird_annotated.jpg', image)

    return {"objects": detected_objects}
