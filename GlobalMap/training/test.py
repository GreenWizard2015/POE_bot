import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

from GlobalMap.training.CGlobalMapGenerator import CDataGenerator
from GlobalMap.training.CGlobalMapDefaultModel import CGlobalMapDefaultModel

import cv2
import numpy as np

gen = CDataGenerator(
  folder='dataset/',
  batchSize=5,
  batchesPerEpoch=1,
  seed=0,
  bigMapSize=1024,
  minCommonPoints=20,
  minInnerPoints=80,
  smallMapSize=256
)

# model = CGlobalMapDefaultModel()
# model.network.summary()
# model.load(only_fully_trained=True)

while True:
  gen.on_epoch_end()
  samplesIn, samplesOut = gen[0]
  for (A, B, ytrue) in zip(*samplesIn, samplesOut):
#     pred = model.network.predict([np.array([src]), np.array([crop])])[0]
#     pred = np.where(.8 < pred, 128, 0).astype(np.uint8)
    
    cv2.imshow('A', A)
    cv2.imshow('B', B)
    print(np.array(ytrue) * 255)
    cv2.waitKey()
 