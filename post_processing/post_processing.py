from PIL import Image
import numpy as np
import cv2

# Paths for the pre-trained face detection model and cartoon image
PROTOTXT_PATH = '/Users/htc/Desktop/BrownU/CS1430_Projects/csci1430-final-project/post_processing/deploy.prototxt'
MODEL_PATH = '/Users/htc/Desktop/BrownU/CS1430_Projects/csci1430-final-project/post_processing/res10_300x300_ssd_iter_140000.caffemodel'
CARTOON_PATH = '/Users/htc/Desktop/BrownU/CS1430_Projects/csci1430-final-project/post_processing/face_cartoon.png'


def verify_face(image_path):
    """
    Verifies if the input image contains a face and assigns a confidence score.

    Parameters:
    - image_path: Path to the input image.

    Returns:
    - face_detected: Boolean indicating if a face was detected.
    - confidence_score: Confidence score for the detected face (if any).
    """
    # Load the pre-trained DNN face detector from OpenCV
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return False, 0.0

    # Prepare the image for the DNN face detector
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Run the face detector
    face_net.setInput(blob)
    detections = face_net.forward()

    # Iterate through detections and calculate the confidence score
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Consider it a face if confidence > 0.5
            #print(f"Face detected with confidence score: {confidence:.2f}")
            return True, confidence

    # No face detected
    #print("No face detected.")
    return False, 0.0


def process_image(image_path, output_path):
    """
    Processes the image: removes facial features, detects the face contour,
    and replaces it with a cartoon face.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the final image.
    """
    image = cv2.imread(image_path)
    if image is None:
        #print(f"Error: Unable to load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        #print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(face_roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour + [x, y]
            cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=-1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    if largest_contour is None:
        #print("No face contour detected.")
        return

    x, y, w, h = cv2.boundingRect(largest_contour)
    cartoon = cv2.imread(CARTOON_PATH, cv2.IMREAD_UNCHANGED)
    if cartoon is None:
        #print(f"Error: Unable to load cartoon image from {CARTOON_PATH}")
        return

    scale_factor = 0.7
    cartoon_h, cartoon_w = cartoon.shape[:2]
    aspect_ratio = cartoon_w / cartoon_h
    new_w = int(w * scale_factor) if cartoon_w > cartoon_h else int(h * scale_factor * aspect_ratio)
    new_h = int(new_w / aspect_ratio) if cartoon_w > cartoon_h else int(h * scale_factor)

    resized_cartoon = cv2.resize(cartoon, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset, y_offset = x + (w - new_w) // 2, y + (h - new_h) // 2

    for i in range(resized_cartoon.shape[0]):
        for j in range(resized_cartoon.shape[1]):
            if resized_cartoon[i, j][3] > 0:
                image[y_offset + i, x_offset + j] = resized_cartoon[i, j][:3]

    cv2.imwrite(output_path, image)
    #print(f"Cartoon face added and saved at {output_path}")


def post_process_image(image_path, output_path):
    """
    Apply post-processing to simplify the rendered image by removing small isolated dots or clusters.
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    _, binary_img = cv2.threshold(img_array, 200, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    size_threshold = 100
    filtered_img = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > size_threshold:
            filtered_img[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    smoothed_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)
    final_img = Image.fromarray(smoothed_img)
    final_img.save(output_path)
    #print(f"Post-processed image saved at {output_path}")
