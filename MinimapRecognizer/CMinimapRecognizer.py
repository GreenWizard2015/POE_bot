import numpy as np
from training.CmrDefaultModel import CmrDefaultModel

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
  def __init__(self, threshold=0.85, model=None):
    self._threshold = threshold
    self._model = model
    if not model:
      self._model = CmrDefaultModel()
      self._model.load(only_fully_trained=True)
    
    self._dims = self._model.input_shape[:2]
    
    self._regions = splitRegions(256, 256, *self._dims)
    self._overlaps = np.zeros((256, 256))
    w, h = self._dims
    for x, y in self._regions:
      self._overlaps[x:x+w, y:y+h] += np.ones(self._dims)

    pass
  
  def process(self, minimap):
    minimap = self._model.preprocessing(minimap)
    w, h = self._dims
    predictions = self._model.predict(
      np.array([ minimap[x:x+w, y:y+h] for x, y in self._regions ])
    )

    maskA = np.zeros((256, 256))
    maskB = np.zeros((256, 256))
    for i, (x, y) in enumerate(self._regions):
      predicted = predictions[i]
      maskA[x:x+w, y:y+h] += predicted[:, :, 0]
      maskB[x:x+w, y:y+h] += predicted[:, :, 1]

    maskA /= self._overlaps
    maskB /= self._overlaps
    
    maskA = np.where(self._threshold < maskA, 255, 0).astype(np.uint8)
    maskB = np.where(self._threshold < maskB, 255, 0).astype(np.uint8)
    return maskA, maskB