import sys
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))

import glob
import shutil
from MinimapRecognizer.CMinimapRecognizer import CMinimapRecognizer
from GlobalMap.training.findGlobalShift import findGlobalShift
import cv2
import numpy as np

recognizer = CMinimapRecognizer()
datasetFolder = os.path.join(os.path.dirname(__file__), 'dataset')
raw = os.path.join(datasetFolder, 'raw')

areas = set()
for p in glob.glob(os.path.join(raw, '*')):
  # copy /dataset/raw/**/*.* into /dataset/**/*.* 
  print('Processing /%s...' % os.path.basename(p))
  dest = os.path.join(datasetFolder, os.path.basename(p))
  shutil.rmtree(dest, ignore_errors=True)
  shutil.copytree(p, dest)


  ###############
  print('Creating masks...')
  for src in glob.glob(os.path.join(dest, '*/*_input.jpg')):
    areas.add(os.path.dirname(src))
    walls, unknown = recognizer.process(cv2.imread(src))
     
    fn = lambda x: src.replace('_input.jpg', '_%s.jpg' % x)
    cv2.imwrite(fn('walls'), walls)
    cv2.imwrite(fn('unknown'), unknown)
  ###############
  
for area in areas:
  print('Processing %s' % area)
  walls = sorted(
    glob.glob(os.path.join(area, '*_walls.jpg'))
  )

  print('Calculate shifts...')
  shifts = [(0, 0)]
  for AFile, BFile in zip(walls[:-1], walls[1:]):
    A = cv2.imread(AFile, cv2.IMREAD_GRAYSCALE)
    A[np.where(A < 80)] = 0
    
    B = cv2.imread(BFile, cv2.IMREAD_GRAYSCALE)
    B[np.where(B < 80)] = 0
    
    shifts.append(findGlobalShift(A, B))
  
  # calc total size
  x1 = y1 = 0
  x2 = y2 = 0
  x = y = 0
  for X, Y in shifts:
    x += X
    y += Y
    
    x1 = min((x1, x))
    y1 = min((y1, y))
    
    x2 = max((x2, x + 256))
    y2 = max((y2, y + 256))
  
  # generate global map (walls)
  x = -x1
  y = -y1
  
  gm = np.zeros((x2 - x1, y2 - y1))
  for AFile, (shiftX, shiftY) in zip(walls, shifts):
    x += shiftX
    y += shiftY
    A = cv2.imread(AFile, cv2.IMREAD_GRAYSCALE)
    gm[x:x+256, y:y+256] += np.where(80 < A, 1, 0)
  
  gm = np.where(0 < gm, 255, 0).astype(np.uint8)
  cv2.imwrite(os.path.join(area, 'global_walls.jpg'), gm)