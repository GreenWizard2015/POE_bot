import tensorflow as tf
from training.NNDebugCallback import NNDebugCallback
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
)

from model import MRModel, model_hash, StackedMRModel
from training.DataGenerator import CDataGenerator
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import time

def dice_coef(y_true, y_pred, eps=0.00001):
  # (batch, h, w, classes) -> (batch, classes, h, w) 
  y_true = K.permute_dimensions(y_true, (0, 3, 1, 2))
  y_pred = K.permute_dimensions(y_pred, (0, 3, 1, 2))
  
  axis = [2, 3]
  intersection = K.sum(y_pred * y_true, axis=axis)
  sums = (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis))

  # standard dice
  dice = (2. * intersection + eps) / (sums + eps)
  # weights per class per sample
#   weights = [.0, .4, .6]
#   dice = K.sum(dice * weights, axis=-1)
  # mean over all samples in the batch
  return K.mean(dice, axis=-1)

def MRModelLoss(y_true, y_pred):
  return 1. - dice_coef(y_true, y_pred)

batch_size = 32
dims = (64, 64)

model, modelA, modelB = StackedMRModel((*dims, 3), [MRModel((*dims, 3), 2), MRModel((*dims, 3), 2)])
weightsFile = '../stacked.h5' # % (model_hash(model), *dims)
if os.path.exists(weightsFile):
  model.load_weights(weightsFile)
else:
  modelA.load_weights('../modelA.h5')
  modelB.load_weights('../modelB.h5')

for submodel in [modelA, modelB]:
  submodel.trainable = False
  for layer in submodel.layers: layer.trainable = False

model.compile(
  optimizer=keras.optimizers.Adam(lr=0.001),
  loss=MRModelLoss,
  metrics=[]
)

seed = time.time_ns() # kinda random seed
dataGen = lambda x, bpe: CDataGenerator(x, dims=dims, batchSize=batch_size, batchesPerEpoch=bpe, seed=seed)
model.fit(
  x=dataGen('dataset/train', 16*4),
  validation_data=dataGen('dataset/validation', 4*4),
  verbose=2,
  callbacks=[
    keras.callbacks.ModelCheckpoint(
      filepath=weightsFile,
      save_weights_only=True,
      save_best_only=True,
      monitor='val_loss',
      mode='min',
      verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss', factor=0.8,
      patience=50
    ),
    NNDebugCallback(
      dims=dims,
      saveFreq=-1,
      dstFolder='debug/%d' % time.time_ns(),
      inputs=(
        cv2.imread('debug/src_input.jpg'),
        cv2.imread('debug/src_walls.jpg', cv2.IMREAD_GRAYSCALE),
        cv2.imread('debug/src_unknown.jpg', cv2.IMREAD_GRAYSCALE),
      )
    )
  ],
  epochs=15000
)

model.save_weights(
  '../weights_%s_%d_%d_latest.h5' % (model_hash(model), *dims)
)