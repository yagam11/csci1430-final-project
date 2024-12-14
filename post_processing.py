from PIL import Image
import numpy as np
import cv2

def post_process_image(image_path, output_path):
    """
    Apply post-processing to simplify the rendered image by removing small isolated dots or clusters.
    :param image_path: Path to the input image.
    :param output_path: Path to save the processed image.
    """
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Step 1: Binarization (Thresholding)
    _, binary_img = cv2.threshold(img_array, 200, 255, cv2.THRESH_BINARY)  # High threshold for clean binarization

    # Step 2: Connected Component Analysis to find and remove small dots
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    # Create a mask to retain only significant components
    size_threshold = 100  # Adjust based on the size of 'dots' to remove
    filtered_img = np.zeros_like(binary_img)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > size_threshold:  # Keep components larger than the threshold
            filtered_img[labels == i] = 255

    # Step 3: Optionally apply morphological closing to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    smoothed_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)

    # Step 4: Convert back to PIL image and save
    final_img = Image.fromarray(smoothed_img)
    final_img.save(output_path)
    print(f"Post-processed image saved at {output_path}")
