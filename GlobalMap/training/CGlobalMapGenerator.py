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
    
    crops = []
    while len(crops) < N:
      overlapArea = self._random.randint(10, self._smallMapSize // 5)
      
      minSz = 2 * overlapArea + self._smallMapSize
      X_w = min((self._bigMapSize, self._random.randint(minSz, w - minSz)))
      X_h = min((self._bigMapSize, self._random.randint(minSz, h - minSz)))
      
      # LT of X
      X_x = self._random.randint(0, w - X_w)
      X_y = self._random.randint(0, h - X_h)
      
      minimapX = self._random.randint(0, X_w - self._smallMapSize)
      minimapY = self._random.randint(0, X_h - self._smallMapSize)
      
      minimap = img[
        (X_x + minimapX):(X_x + minimapX + self._smallMapSize),
        (X_y + minimapY):(X_y + minimapY + self._smallMapSize)
      ]
      # check overlap
      overlap = minimap.copy()
      overlap[overlapArea:-overlapArea, overlapArea:-overlapArea] = 0
      innerPoints = np.count_nonzero(overlap)
      if innerPoints < self._minCommonPoints: continue
      if (np.count_nonzero(minimap) - innerPoints) < self._minInnerPoints: continue
      #
      bigMap = np.zeros((self._bigMapSize, self._bigMapSize))
      B_x = self._random.randint(0, self._bigMapSize - X_w)
      B_y = self._random.randint(0, self._bigMapSize - X_h)
      
      bigMap[B_x:B_x+X_w, B_y:B_y+X_h] = img[X_x:(X_x + X_w), X_y:(X_y + X_h)]
      bigMap[
        (B_x + minimapX):(B_x + minimapX + self._smallMapSize),
        (B_y + minimapY):(B_y + minimapY + self._smallMapSize)
      ] = overlap

      crops.append((
        bigMap,
        minimap,
        (
          B_x + minimapX + self._smallMapSize / 2,
          B_y + minimapY + self._smallMapSize / 2
        )
      ))

    return crops
    
  def __getitem__(self, index):
    sampleWalls, _ = self._batchData( self._epochBatches[index] )
    crops = self._generateCrops(sampleWalls)
    Y = np.array([self._asOHE(x[2]) for x in crops])
    return (
      (
        np.array([x[0] for x in crops]),
        np.array([x[1] for x in crops])
      ),
      (
        Y[:, 0], Y[:, 1]
      )
    )
  ###########################
  def _loadMasks(self, srcWalls):
    imgWalls = cv2.imread(srcWalls, cv2.IMREAD_GRAYSCALE)
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    return [imgWalls]
  
  def _asOHE(self, pos):
    res = np.zeros((2, self._bigMapSize), np.uint8)
    for i, p in enumerate(pos):
      res[i, int(p)] = 1
    return res
