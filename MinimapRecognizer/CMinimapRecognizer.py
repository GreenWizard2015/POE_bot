from model import MRModel, model_hash, StackedMRModel
import numpy as np
import os
import cv2

def splitRegions(h, w, rh, rw):
  regions = []
  for x in range(0, 1 + w - rw, rw):
    for y in range(0, 1 + h - rh, rh):
      regions.append((x, y))
  
  if (0 < (w % rw)) or (0 < (h % rh)):
    for pt in ((w - rw, 0), (w - rw, h - rh)):
      if pt not in regions:
        regions.append(pt)

  return regions

class CMinimapRecognizer:
  def __init__(self, threshold=0.01, dims=None, model=None):
    self._threshold = threshold
    self._dims = dims if dims else (64, 64)
    
    self._model = model
    if not model:
      dims = self._dims
      self._model = StackedMRModel((*dims, 3), [MRModel((*dims, 3), 2), MRModel((*dims, 3), 2)])[0]
      self._model.load_weights(
        '%s/stacked.h5' % (os.path.dirname(__file__))
      )
    
    self._regions = splitRegions(256, 256, *self._dims)
    self._overlaps = np.zeros((256, 256))
    w, h = self._dims
    for x, y in self._regions:
      self._overlaps[x:x+w, y:y+h] += np.ones(self._dims)

    pass
  
  def process(self, minimap):
    minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2Lab) / 255. # normalize
    w, h = self._dims
    predictions = self._model.predict(
      np.array([ minimap[x:x+w, y:y+h] for x, y in self._regions ])
    )

    maskA = np.zeros((256, 256))
    maskB = np.zeros((256, 256))
    for i, (x, y) in enumerate(self._regions):
      predicted = predictions[i, :, :, -2:]
      maskA[x:x+w, y:y+h] += predicted[:, :, 0]
      maskB[x:x+w, y:y+h] += predicted[:, :, 1]

    maskA /= self._overlaps
    maskB /= self._overlaps
    
    maskA = np.where(self._threshold < maskA, 255, 0).astype(np.uint8)
    maskB = np.where(self._threshold < maskB, 255, 0).astype(np.uint8)
    return maskA, maskB