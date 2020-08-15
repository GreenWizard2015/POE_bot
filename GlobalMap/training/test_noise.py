import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

from GlobalMap.training.CNoiseMapGenerator import CNoiseMapGenerator

import cv2

gen = CNoiseMapGenerator(
  folder='dataset/',
  batchSize=5,
  batchesPerEpoch=1,
  seed=0,
  outputMapSize=64,
  noiseAdded=64,
  noiseVanished=64
)

from GlobalMap.training.CMapKeypointsModel import CMapKeypointsModel

model = CMapKeypointsModel()
model.load(only_fully_trained = True)
model.network.summary()

while True:
  gen.on_epoch_end()
  samplesIn, samplesOut = gen[0]
  for (A, B) in zip(samplesIn, samplesOut):
    cv2.imshow('A', A)
    cv2.imshow('B', cv2.resize(B[0], (256, 256)))

    prediction = model.network.predict(A.reshape((1, 256, 256, 1)))[0]
    diff = B[0] - prediction[:, :, 0]
    cv2.imshow('pred', cv2.resize(diff, (256, 256)))
    cv2.waitKey()
