import sys
import os
import tensorflow as tf

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(__file__)))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5 * 1024)]
  )

from training.commonTrainingLoop import commonTrainingLoop
from training.CmrModelA import CmrModelA
from training.CmrModelB import CmrModelB
from training.CmrDefaultModel import CmrDefaultModel

def trainModel(model):
  model.load(only_fully_trained = False, reset=True)
  commonTrainingLoop(
    model,
    batch_size=32,
    batch_per_epoch=32,
    batch_per_validation=0.2
  )
  return

########################
# train all NN from zero
trainModel(CmrModelA())
trainModel(CmrModelB())
trainModel(CmrDefaultModel())