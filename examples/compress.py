import cv2
import os

def compress_image_by_half(image_path):
    """
    Compress a specific image by half and save the output in the same folder.
    :param image_path: Path to the image file.
    """
    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to read the image '{image_path}'.")
        return

    # Get new dimensions (50% of original size)
    height, width = image.shape[:2]
    new_dimensions = (width // 2, height // 2)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    # Get the output path (same folder as the original image)
    folder, file_name = os.path.split(image_path)
    file_name_without_ext, ext = os.path.splitext(file_name)
    output_path = os.path.join(folder, f"{file_name_without_ext}_compressed{ext}")

    # Save the resized image
    cv2.imwrite(output_path, resized_image)

    print(f"Compressed image saved to '{output_path}'.")

# Example usage
if __name__ == "__main__":
    # Replace this with the path to your specific image
    image_path = "/Users/htc/Desktop/BrownU/CS1430_Projects/csci1430-final-project/examples/IMG_6196.jpg"
    compress_image_by_half(image_path)
