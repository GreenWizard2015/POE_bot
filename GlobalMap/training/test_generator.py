import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

from GlobalMap.training.CDataGenerator import CDataGenerator
from GlobalMap.training.CNoiseInputGenerator import CNoiseInputGenerator
from GlobalMap.training.CNavigatorModel import CNavigatorModel

import cv2
import numpy as np

gen = CNoiseInputGenerator(
  gen=CDataGenerator(
    'dataset/validation', 
    batchSize=5, batchesPerEpoch=1, seed=0,
    useCrop=True
  ),
  seed=0,
  noiseAdded=64,
  noiseVanished=64
)

model = CNavigatorModel()
model.network.summary()
model.load(only_fully_trained=True)

while True:
  gen.on_epoch_end()
  samplesIn, samplesOut = gen[0]
  for (A, B) in zip(samplesIn, samplesOut):
    pred = model.predict(A)
    
    res = np.dstack((
      B[0],
      pred[:, :, 0] * 255,
      np.zeros_like(B[0])
    ))
    
    cv2.imshow('A', A)
    cv2.imshow('B', res)
    
    cv2.waitKey()
