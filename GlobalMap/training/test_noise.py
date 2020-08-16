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
  outputMapSize=256,
  noiseAdded=64,
  noiseVanished=64
)

from GlobalMap.training.CMapKeypointsModel import CMapKeypointsModel
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def compressorModel():
  model = CMapKeypointsModel()
  model.load(only_fully_trained = True)
  model.network.summary()
  
  return keras.Model(
    inputs=model.network.input,
    outputs=[model.compressed(), model.network.output]
  )

model = compressorModel()

while True:
  gen.on_epoch_end()
  samplesIn, samplesOut = gen[0]
  for (A, B) in zip(samplesIn, samplesOut):
    cv2.imshow('input', A)
    cv2.imshow(
      'simple downscale',
      cv2.resize(cv2.resize(A, (64, 64)), (256, 256))
    )

    (prediction, ), (reconstructed, ) = model.predict(A.reshape((1, 256, 256, 1)))
    prediction = prediction.sum(axis=-1)
    prediction -= prediction.min()
    cv2.imshow('compressed', cv2.resize(prediction / prediction.max(), (256, 256)))
    
    cv2.imshow('reconstructed', reconstructed)
    cv2.waitKey()
