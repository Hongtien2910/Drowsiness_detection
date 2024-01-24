import os
import sys
import glob
import time
import dlib
import multiprocessing


faces_folder = "D:/PBL4/Drowsiness-Detection/train/images"

options = dlib.shape_predictor_training_options()


options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 15
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_amount = 5
options.oversampling_translation_jitter = 0.1
options.be_verbose = True
options.oversampling_amount = 300

# Define the number of CPUs cores to be used when training
options.num_threads = multiprocessing.cpu_count()

training_xml_path = os.path.join(faces_folder, "D:/PBL4/Drowsiness-Detection/train/data.xml")

# Đo thời gian bắt đầu
start_time = time.time()

# Bắt đầu quá trình huấn luyện
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

# Đo thời gian kết thúc
end_time = time.time()

# In ra thời gian chạy huấn luyện và thời gian còn lại để hoàn thành
elapsed_time = end_time - start_time
print(f"\nTraining completed in {elapsed_time} seconds")
