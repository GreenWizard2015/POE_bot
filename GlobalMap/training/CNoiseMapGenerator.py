import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob
import os

class CNoiseMapGenerator(Sequence):
  def __init__(self, folder, 
    batchSize, batchesPerEpoch, seed,
    outputMapSize, noiseAdded, noiseVanished
  ):
    self._batchSize = batchSize
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    
    self._outputMapSize = outputMapSize
    self._minAdded = noiseAdded
    self._minVanished = noiseVanished
    
    self._epochBatches = None
    self._images = [
      ( 
        self._loadMasks(f),
      ) for f in glob('%s/**/*_walls.jpg' % folder, recursive=True) if self._validMask(f)
    ]
    self.on_epoch_end()
    return

  def __len__(self):
    return self._batchesPerEpoch

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epochBatches = self._random.choices(self._images, k=self._batchesPerEpoch)
    return
  
  def _batchData(self, data):
    sampleWalls = data[0][0]
    return (sampleWalls, None)
    
  def _generateSamples(self, img, N=None):
    N = N if N else self._batchSize

    cleanedMap = np.square(cv2.resize(img * 255, (self._outputMapSize, self._outputMapSize)) / 255.0)
    cleanedMap[np.where(0 < cleanedMap)] = 1
    cleanedMap = cleanedMap.reshape((1, self._outputMapSize, self._outputMapSize))
    res = []
    while len(res) < N:
      mapInput = img.copy()
      added = vanished = 0
      while (added < self._minAdded) or (vanished < self._minVanished):
        x = self._random.randint(0, img.shape[0] - 3)
        y = self._random.randint(0, img.shape[0] - 3)
        area = mapInput[x:x+3, y:y+3]
        total = np.sum(area)
        
        if (0 < area[1, 1]) and (vanished < self._minVanished) and (3 < total):
          mapInput[x + 1, y + 1] = 0 
          vanished += 1
          continue
        
        if (area[1, 1] < 1) and (added < self._minAdded) and (total < 5):
          mapInput[x + 1, y + 1] = 1
          added += 1
          continue
      #
      res.append((
        mapInput,
        cleanedMap
      ))

    return res
    
  def __getitem__(self, index):
    sampleWalls, _ = self._batchData( self._epochBatches[index] )
    samples = self._generateSamples(sampleWalls)
    return (
      np.array([x[0] for x in samples]),
      np.array([x[1] for x in samples])
    )
  ###########################
  def _loadMasks(self, srcWalls):
    imgWalls = cv2.imread(srcWalls, cv2.IMREAD_GRAYSCALE)
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    return [imgWalls]
  
  def _validMask(self, f):
    return not os.path.basename(f).startswith('global')