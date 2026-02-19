import cv2
from ultralytics import YOLO

id_card_model = YOLO('ID_detection.pt')
line_detection_model = YOLO('line_detection.pt')

# Load the image
image = cv2.imread('EgyptianID-Detection-1/test/images/7_jpg.rf.2ab8f897048ac26d68f700a409323690.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
id_results = id_card_model(image)

# Extract the ID card bounding box
for box in id_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_id_card = image[y1:y2, x1:x2]

    # Step 2: Detect text lines within the ID card
    line_results = line_detection_model(cropped_id_card)

    # Crop and save each detected line
    for i, line_box in enumerate(line_results[0].boxes):
        lx1, ly1, lx2, ly2 = map(int, line_box.xyxy[0])
        cropped_line = cropped_id_card[ly1:ly2, lx1:lx2]
        cv2.imwrite(f'line_{i}.png', cropped_line)