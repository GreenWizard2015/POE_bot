import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

import os
import glob
import shutil
from CMinimapRecognizer import CMinimapRecognizer
import cv2

recognizer = CMinimapRecognizer()
datasetFolder = os.path.join(os.path.dirname(__file__), 'dataset')
raw = os.path.join(datasetFolder, 'raw')

for p in glob.glob(os.path.join(raw, '*')):
  # copy /dataset/raw/**/*.* into /dataset/**/*.* 
  print('Processing /%s...' % os.path.basename(p))
  dest = os.path.join(datasetFolder, os.path.basename(p))
  shutil.rmtree(dest, ignore_errors=True)
  shutil.copytree(p, dest)
  
  ###############
  print('Creating masks...')
  for src in glob.glob(os.path.join(dest, '*/*_input.jpg'), recursive=not True):
    walls, unknown = recognizer.process(cv2.imread(src))
    
    fn = lambda x: src.replace('_input.jpg', '_%s.jpg' % x)
    cv2.imwrite(fn('walls'), walls)
    cv2.imwrite(fn('unknown'), unknown)
  ###############
  # TODO: Combine [i]_*.jpg and [i + 1]_*.jpg into [i]_[i + 1]_*.jpg
  print('Creating global masks...')
  print()
  
