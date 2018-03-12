import os
import sys
import glob

import dlib
from skimage import io

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 5
options.num_threads = 4
options.be_verbose = True
options.upsample_limit = 1
# options.detection_window_size = 64*128

dlib.train_simple_object_detector('Dataprep/annotations.xml', 'dlib.svm', options)
