from model import MRModel, model_hash
from training.DataGenerator import CDataGenerator
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import tensorflow as tf
import cv2
import numpy
import hashlib
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
          gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def MRModelLoss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

batch_size = 32
dims = (240, 240)

model = MRModel((*dims, 3), 2)
model.compile(
  optimizer=keras.optimizers.Adam(lr=0.005),
  loss=MRModelLoss,
  metrics=[]
)

weightsFile = '../weights_%s_%d_%d.h5' % (model_hash(model), *dims)
if os.path.exists(weightsFile):
  model.load_weights(weightsFile)

seed = time.time_ns() # kinda random seed
dataGen = lambda x: CDataGenerator(x, dims=dims, batchSize=batch_size, batchesPerEpoch=10, seed=seed)
model.fit(
  x=dataGen('dataset/train'),
  validation_data=dataGen('dataset/validation'),
  verbose=2,
  callbacks=[
    keras.callbacks.ModelCheckpoint(
      filepath=weightsFile,
      save_weights_only=True,
      save_best_only=True,
      monitor='val_loss',
      mode='min',
      verbose=1
    )
  ],
  epochs=250
)

######
# debug
model = MRModel((*dims, 3), 2)
if os.path.exists(weightsFile):
  model.load_weights(weightsFile)

testSet = CDataGenerator('dataset/train', dims=dims, batchSize=5, batchesPerEpoch=1)
X, Y = testSet[1]
for img, mask in list(zip(X, Y)):
  cv2.imshow('original', cv2.resize(img, (640, 480)))
  mask1 = mask[:,  :, 0]
  cv2.imshow('mask 1', cv2.resize(mask1, (640, 480)))
  
  predict = model.predict(numpy.array([img]))[0]
  predict = numpy.where(0.5 < predict, 1., 0)
  cv2.imshow('predict', cv2.resize(predict[:, :, 0], (640, 480)))
  cv2.waitKey(0)
######