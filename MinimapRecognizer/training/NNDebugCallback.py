import cv2
import os
import math
import numpy as np
import tensorflow.keras as keras
from CMinimapRecognizer import CMinimapRecognizer

def diffMap(predict, truth):
  res = np.zeros((*truth.shape[:2], 3), np.uint8)
  truth = 50 < truth
  predict = 50 < predict

  res[np.where(predict)] = [0, 0, 255]
  res[np.where(truth)] = [127,127,127]
  res[np.where(truth & predict)] = [0, 255, 0]
  return res

class NNDebugCallback(keras.callbacks.Callback):
  def __init__(self, dims, saveFreq, dstFolder, inputs):
    self._dims = dims
    self._best = math.inf
    self._saveFreq = saveFreq
    self._dstFolder = dstFolder
    os.makedirs(dstFolder, exist_ok=True)
    self._src, self._walls, self._unknown = inputs
    pass
  
  def on_epoch_end(self, epoch, logs=None):
    loss = logs['val_loss']
    sheduled = (0 < self._saveFreq) and (0 == (epoch % self._saveFreq)) 
    if (self._best < loss) and not sheduled: return
    
    self._best = min((self._best, loss))
    self.dump('e%06d' % epoch)
  
  def on_train_end(self, logs=None):
    self.dump('last')
    
  def dump(self, epoch):
    dest = lambda x: os.path.join(self._dstFolder, '%s_%s.jpg' % (epoch, x))
    
    recognizer = CMinimapRecognizer(threshold=0.1, dims=self._dims, model=self.model)
    walls, unknown = recognizer.process(self._src)
    
    cv2.imwrite(dest('walls'), diffMap(walls, self._walls))
    cv2.imwrite(dest('unknown'), diffMap(unknown, self._unknown))
    return
