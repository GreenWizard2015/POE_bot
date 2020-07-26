import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob
from math import floor

'''
  ---------------------------------
  -                               -
  -   X------------------------|  -
  -   |       |                |  -
  -   |   B********************|  -
  -   |   *   |                |  -
  -   |---*---S                |  -
  -   |   *                    |  -
  -   |------------------------|  -
  -                               -
  ---------------------------------
  
  - Take masks
  - Add padding (wh -> 3*wh)
  - Take a random region X and split into S and B
    - X must contain >xMin 'pixels' 
    - area SB must contain >overlapMin 'pixels'
  
  - Return (big map), (small map), (x, y of S in B)
'''
class CDataGenerator(Sequence):
  def __init__(self, folder, 
    batchSize, batchesPerEpoch, seed,
    bigMapSize, smallMapSize,
    minCommonPoints, minInnerPoints
  ):
    self._minInnerPoints = minInnerPoints
    self._minCommonPoints = minCommonPoints
    self._batchSize = batchSize
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    
    self._bigMapSize = bigMapSize
    self._smallMapSize = smallMapSize
    
    self._epochBatches = None
    self._images = [
      ( 
        self._loadMasks(f),
      ) for f in glob('%s/**/global_walls.jpg' % folder, recursive=True)
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
    
  def _generateCrops(self, img, N=None):
    N = N if N else self._batchSize
    w = img.shape[0]
    h = img.shape[1]
    
    hsz = self._smallMapSize // 2
    crops = []
    while len(crops) < N:
      overlapArea = self._random.randint(10, hsz)
      X_w = X_h = 2 * self._smallMapSize
      
      X_x = self._random.randint(0, w - X_w) + hsz
      X_y = self._random.randint(0, h - X_h) + hsz
      
      while True:
        map_dX = self._random.randint(-hsz, hsz)
        if 5 < abs(map_dX): break
      
      while True:
        map_dY = self._random.randint(-hsz, hsz)
        if 5 < abs(map_dY): break
      
      mapA = img[
        (X_x):(X_x + self._smallMapSize),
        (X_y):(X_y + self._smallMapSize)
      ]
      mapB = img[
        (X_x + map_dX):(X_x + map_dX + self._smallMapSize),
        (X_y + map_dY):(X_y + map_dY + self._smallMapSize)
      ]
      # check overlap
      innerPoints = np.count_nonzero(mapA) + np.count_nonzero(mapB)
      if innerPoints < self._minCommonPoints: continue
      #
      crops.append((
        mapA,
        mapB,
        [
          (map_dX - hsz) / self._smallMapSize,
          (map_dY - hsz) / self._smallMapSize
        ]
      ))

    return crops
    
  def __getitem__(self, index):
    sampleWalls, _ = self._batchData( self._epochBatches[index] )
    crops = self._generateCrops(sampleWalls)
    return (
      (
        np.array([x[0] for x in crops]),
        np.array([x[1] for x in crops])
      ),
      np.array([x[2] for x in crops])
    )
  ###########################
  def _loadMasks(self, srcWalls):
    imgWalls = cv2.imread(srcWalls, cv2.IMREAD_GRAYSCALE)
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    return [imgWalls]
