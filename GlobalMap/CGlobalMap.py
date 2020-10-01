import numpy as np
import cv2
import math
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra
import skimage.measure

def distancesMap(img, startPt):
  img = np.pad(img, 1)
  def to_index(y, x): return y * img.shape[1] + x
  def to_coordinates(index): return index // img.shape[1], index % img.shape[1]

  adjacency = dok_matrix((img.shape[0] * img.shape[1], img.shape[0] * img.shape[1]), dtype=bool)
  
  # The following lines fills the adjacency matrix by
  directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
  for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):
      if img[i, j] <= 0: continue
  
      for y_diff, x_diff in directions:
        if 0 < img[i + y_diff, j + x_diff]:
          adjacency[to_index(i, j), to_index(i + y_diff, j + x_diff)] = True
  
  # Compute the shortest path between the source and all other points in the image
  m = dijkstra(
    adjacency, indices=[to_index(1 + startPt[1], 1 + startPt[0])],
    directed=False, unweighted=True, return_predecessors=False
  )
  
  dst = np.zeros_like(img, np.float)
  for ind, d in enumerate(m[0]):
    i,j = to_coordinates(ind) 
    dst[i,j] = d
  return dst[1:-1, 1:-1]

def findShift(A, B):
  B = cv2.copyMakeBorder(B, *A.shape[:2], *A.shape[:2], borderType=cv2.BORDER_CONSTANT, value=0)
  
  match = cv2.matchTemplate(B, A, cv2.TM_CCORR_NORMED)
  
  ptsX, ptsY  = np.unravel_index(np.argmax(match), match.shape)
  anchor = (np.array(B.shape[:2]) - np.array(A.shape[:2])) // 2
  return anchor - np.array([ptsX, ptsY])

class CGlobalMap:
  def __init__(self):
    self._map = np.zeros((2, 1024, 1024), np.float32)
    self._prev = None
    self._prevPos = np.array(self._map.shape[1:]) * 2
    self._mapArea = [math.inf, math.inf, -math.inf, -math.inf]
    self._pos = np.zeros((2,), np.int32)
    self.FORGET_RATE = 0.5
    return
  
  def update(self, maskWalls, maskUnknown):
    shift = np.zeros(2) if self._prev is None else findShift(self._prev, maskWalls)

    pos = self._prevPos + shift
    x, y = pos.astype(np.int32) // 4

    dim = maskWalls.shape[0] // 4
    self._mapArea = [
      min((self._mapArea[0], x)),
      min((self._mapArea[1], y)),
      max((self._mapArea[2], x + dim)),
      max((self._mapArea[3], y + dim)),
    ]
    
    oldArea = self._map[:, x:x+dim, y:y+dim]
    masks = np.array([
      skimage.measure.block_reduce(maskWalls, (4, 4), np.max),
      skimage.measure.block_reduce(maskUnknown, (4, 4), np.max)
    ])
    self._map[:, x:x+dim, y:y+dim] = (oldArea + self.FORGET_RATE * masks) / (1 + self.FORGET_RATE)
    
    self._pos = (np.array([x, y]) + np.array([dim, dim]) / 2).astype(np.int32)
    self._prev = maskWalls
    self._prevPos = pos
    return 
  
  def process(self):
    cx, cy = self._pos 
    dim = 64 # 512X512 real map
    area = np.where(100 < self._map[:, cx-dim:cx+dim, cy-dim:cy+dim], 0, 1)
    area[:, dim, dim] = 1 # cutoff center
    DST = distancesMap(area[0], startPt=(dim, dim))

    dst = DST.copy()
    dst[np.where(0 < area[1])] = math.inf # only unknown areas
    ptsX, ptsY = np.unravel_index(np.argmin(dst), dst.shape)
    
    directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
    path = []
    try:
      while (len(path) < 1256) and (0 < DST[ptsX, ptsY]):
        p = np.array([ptsX, ptsY])
        bestScore = math.inf
        bestDir = 0
        for i in range(4):
          x, y = directions[i] + p
          s = DST[x, y]
          if s < bestScore:
            bestScore, bestDir = s, i
            
        ptsX, ptsY = directions[bestDir] + p
        path.append((ptsX, ptsY))
    except:
      pass

    dest = np.array(path[-10:][0])
    
    # visualize path
    dbg = np.zeros((*area.shape[1:], 3))
    dbg[:,:,0] = 1 - area[0]
    dbg[:,:,1] = 1 - area[1]
    for x, y in path:
      dbg[x, y, 2] = 1
    dbg[dest[0], dest[1], :] = 1
    cv2.imshow('gm', cv2.resize(dbg, (645, 645)))
    ###
    return dest - dim