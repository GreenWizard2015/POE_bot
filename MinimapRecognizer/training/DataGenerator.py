import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob

""" select single image per epoch and return random crop of inputDims """
class CDataGenerator(Sequence):
  def __init__(self, folder, dims, batchSize=32, batchesPerEpoch=2, seed=0):
    self._images = [
      (str(f), str(f).replace('_input.jpg', '_mask.jpg')) for f in glob('%s/*_input.jpg' % folder)
    ]
    self._batchSize = batchSize
    self._dims = dims
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    self._epoch = None
    self.on_epoch_end()

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epoch = self._random.choice(self._images)

  def __len__(self):
    return self._batchesPerEpoch

  def __getitem__(self, index):
    # I'm ignoring index, maybe it's bad idea
    sampleInput, sampleMask = self._epoch
    sampleInput = cv2.imread(sampleInput) / 255 # normalize
    sampleMask = cv2.imread(sampleMask, cv2.IMREAD_GRAYSCALE)
    
    cw, ch = self._dims
    w, h = np.array([sampleInput.shape[0] - cw, sampleInput.shape[1] - ch])
    crops = []
    for _ in range(self._batchSize):
      x = self._random.randint(0, w)
      y = self._random.randint(0, h) 
      crops.append((x, y, x + cw, y + ch))

    X = self._generate_X(sampleInput, crops)
    y = self._generate_y(sampleMask, crops)
    return X, y

  def _generate_X(self, img, crops):
    X = np.empty((self._batchSize, *self._dims, 3))

    for i, (x1, y1, x2, y2) in enumerate(crops):
      X[i,] = img[x1:x2, y1:y2]

    return X

  def _generate_y(self, img, crops):
    y = np.empty((self._batchSize, *self._dims, 2))

    for i, (x1, y1, x2, y2) in enumerate(crops):
      crop = img[x1:x2, y1:y2]
      mask = np.empty((2, *self._dims))
      # I saved masks in jpg, so there is an artifacts :)
      mask[0] = (120 < crop) * 1
      mask[1] = (228 < crop) * 1
      y[i,] = mask.transpose(1, 2, 0) # (2, w, h) -> (w, h, 2)

    return y 
