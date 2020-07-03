import tensorflow as tf
import numpy
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

import cv2
import os
import glob
from CMinimapRecognizer import CMinimapRecognizer

srcFolder = '../../minimap/'
dstFolder = os.path.dirname(__file__) + '/output'
os.makedirs(dstFolder, exist_ok=True)
recognizer = CMinimapRecognizer()
for src in glob.glob(srcFolder + '*_input.jpg'):
  filename = os.path.basename(src)
  dest = lambda x: os.path.join(dstFolder, filename.replace('input', x))
  
  img = cv2.imread(src)
  walls, unknown = recognizer.process(img)
  
  cv2.imwrite(dest('input'), img)
  cv2.imwrite(dest('walls'), walls)
  cv2.imwrite(dest('unknown'), unknown)
  
  img[numpy.where(0 < walls)] = [0, 255, 0]
  img[numpy.where(0 < unknown)] = [0, 0, 255]
  cv2.imwrite(dest('masked'), img)
  pass