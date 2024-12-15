import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from post_processing.post_processing import verify_face, process_image, post_process_image  # Import functions

def main():
    opt = TestOptions().parse()
    opt.nThreads = 1   # Test code only supports nThreads = 1
    opt.batchSize = 1  # Test code only supports batchSize = 1
    opt.serial_batches = True  # No shuffle
    opt.no_flip = True  # No flip

    if not os.path.isdir(opt.results_dir):
        os.makedirs(opt.results_dir)

    # Create subdirectories for unprocessed and processed images
    unprocessed_dir = os.path.join(opt.results_dir, "unprocessed")
    processed_dir = os.path.join(opt.results_dir, "processed")

    if not os.path.exists(unprocessed_dir):
        os.makedirs(unprocessed_dir)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # Test and process
    for i, data in enumerate(dataset):
        model.set_input(data)
        img_path = model.get_image_paths()
        model.test()

        result_basename = os.path.splitext(os.path.basename(img_path[0]))[0] + '.png'
        unprocessed_path = os.path.join(unprocessed_dir, f"unprocessed_{result_basename}")
        processed_path = os.path.join(processed_dir, f"processed_{result_basename}")

        # Save the unprocessed image
        model.write_image(opt.results_dir)
        os.rename(os.path.join(opt.results_dir, result_basename), unprocessed_path)

        # Post-process the image
        post_process_image(unprocessed_path, processed_path)

        # Check for a face in the processed image
        face_detected, confidence = verify_face(processed_path)
        if face_detected:
            # Replace the processed image with the cartoon-replaced image
            process_image(processed_path, processed_path)  # Replace the image in-place

if __name__ == '__main__':
    main()
