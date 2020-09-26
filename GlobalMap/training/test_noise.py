import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

import cv2
import numpy as np

from GlobalMap.training.CGlobalMapGenerator import CDataGenerator
from GlobalMap.training.CNoiseInputGenerator import CNoiseInputGenerator
from GlobalMap.training.CMapKeypointsModel import CMapKeypointsModel

folder = lambda x: os.path.join(os.path.dirname(__file__), x)
model = CMapKeypointsModel()
model.load(only_fully_trained=True)

gen = CDataGenerator(
  folder('dataset/train'), 
  batchSize=5, batchesPerEpoch=1, seed=0,
  minCommonPoints=20,
  minInnerPoints=80,
  smallMapSize=256
)

gen = CNoiseInputGenerator(
  gen=gen,
  seed=0,
  outputMapSize=256,
  noiseAdded=64,
  noiseVanished=64
)

while True:
  gen.on_epoch_end()
  samplesIn, samplesOut = gen[0]
  for (A, B, res) in zip(*samplesIn, samplesOut):
    pred = model.predict(A, B)[0]
    print(res, pred)
    cv2.imshow('input A', A)
    cv2.imshow('input B', B)
    cv2.waitKey()
