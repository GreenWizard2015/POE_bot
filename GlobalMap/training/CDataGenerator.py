import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob

from scipy.ndimage.interpolation import rotate

def create_circular_mask(h, w, radius):
  center = (int(w/2), int(h/2))
  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
  return dist_from_center <= radius

class CDataGenerator(Sequence):
  def __init__(self, folder, 
    batchSize, batchesPerEpoch, seed,
    useCrop
  ):
    self._batchSize = batchSize
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    
    self._epochBatches = None
    self._images = [
      f for f in glob('%s/**/*_map.jpg' % folder, recursive=True)
    ]
    self._useCrop = useCrop
    self.on_epoch_end()
    return

  def __len__(self):
    return self._batchesPerEpoch

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epochBatches = self._random.choices(self._images, k=self._batchesPerEpoch)
    return
    
  def __getitem__(self, index):
    f = self._epochBatches[index]
    sampleMap = self._loadMask(f, cv2.IMREAD_COLOR)
    sampleTarget = self._loadMask(f.replace('_map', '_target'), cv2.IMREAD_GRAYSCALE)
    
    samples = []
    for _ in range(self._batchSize):
      X = sampleMap.copy()
      Y = sampleTarget.copy()
      
      angle = int(self._random.random() * 360)
      X = rotate(X, angle, reshape=False)
      Y = rotate(Y, angle, reshape=False)
      
      if self._useCrop:
        forgetRange = .3 + self._random.random()
        mask = create_circular_mask(
          X.shape[0], X.shape[0],
          radius=int((X.shape[0] / 2) * forgetRange)
        )
        X[~mask] = 0
      
      samples.append((X, Y))
    #
    return (
      np.array([x[0] for x in samples]),
      np.array([[x[1], 1 - x[1]] for x in samples])
    )
  ###########################
  def _loadMask(self, src, t):
    img = cv2.imread(src, t)
    img = np.where(80 < img, 1, 0).astype(np.float32)
    return img
