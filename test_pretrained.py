import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from post_processing import post_process_image  # Import the function

def main():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    if not os.path.isdir(opt.results_dir):
        os.makedirs(opt.results_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        img_path = model.get_image_paths()
        print('Processing %04d (%s)' % (i+1, img_path[0]))
        model.test()

        result_basename = os.path.splitext(os.path.basename(img_path[0]))[0] + '.png'
        result_path = os.path.join(opt.results_dir, result_basename)
        model.write_image(opt.results_dir)

        # Call the post-processing function
        processed_path = os.path.join(opt.results_dir, f"processed_{result_basename}")
        post_process_image(result_path, processed_path)
        
if __name__ == '__main__':
    main()
