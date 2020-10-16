import numpy as np
import cv2
import math
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra
import skimage.measure
import time

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
    self._moves = np.zeros((1, 1024, 1024), np.int8)
    self._prev = None
    self._mapArea = [math.inf, math.inf, -math.inf, -math.inf]
    self._pos = None
    self._realPos = np.array(self._map.shape[1:]) * 2 # Half of x4
    self.FORGET_RATE = 0.5
    self._changed = True
    
    ##
    self._collectedFrames = []
    self.FRAMES_LIMIT = 2
    self.FRAMES_SIZE = 128
    self.FRAMES_DIST = int(.6 * self.FRAMES_SIZE / 2) ** 2
    return
  
  def update(self, maskWalls, maskUnknown):
    shift = np.zeros(2)
    self._changed = True
    if not (self._prev is None):
      shift = findShift(self._prev, maskWalls)
      self._changed = 12 < np.sum(np.power(shift, 2)) 
    if not self._changed: return

    self._realPos = (self._realPos + shift).astype(np.int32)
    x, y = self._realPos // 4

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
    
    ######
    x,  y = self._pos
    d = 2
    self._moves[0, x-d:x+d+1, y-d:y+d+1] = 1
    ######
    self._collectFrames(shift // 4)
    return

  def _collectFrames(self, shift):
    dim = self.FRAMES_SIZE // 2
    # update frames
    d = 1
    for frame in self._collectedFrames:
      frame[0] += shift
      x, y = (frame[0] + dim).astype(np.int)
      frame[1][x-d:x+d+1, y-d:y+d+1] = 255

    # remove old frames
    OOR = [i for i, frame in enumerate(self._collectedFrames) if self.FRAMES_DIST < np.sum(np.square(frame[0]))]
    for i in reversed(OOR):
      _, moves, initState = self._collectedFrames[i]
      t = time.time()
      cv2.imwrite('minimap/%d_%d_map.jpg' % (t, i), initState)
      cv2.imwrite('minimap/%d_%d_target.jpg' % (t, i), moves)
      del self._collectedFrames[i]
      
    # add new frame
    if len(self._collectedFrames) < self.FRAMES_LIMIT:
      canAddNew = (0 == len(self._collectedFrames)) or (
        (self.FRAMES_DIST // (self.FRAMES_LIMIT ** 2)) < np.sum(np.square(self._collectedFrames[-1][0]))
      )
      if canAddNew:
        cx, cy = self._pos
        area = np.where(100 < self._map[:, cx-dim:cx+dim, cy-dim:cy+dim], 255, 0)
        moves = self._moves[:, cx-dim:cx+dim, cy-dim:cy+dim] * 255
        self._collectedFrames.append([
          np.zeros((2,)), # current pos
          np.zeros((self.FRAMES_SIZE, self.FRAMES_SIZE)), # future moves
          np.dstack((area[0], area[1], moves[0])) # initially state
        ])
    return
  
  def process(self):
    if not self._changed:
      return np.zeros((2, ))
    
    cx, cy = self._pos 
    dim = 128 # 512X512 real map
    area = np.where(100 < self._map[:, cx-dim:cx+dim, cy-dim:cy+dim], 1, 0)
    area[:, dim, dim] = 0 # cutoff center
    
    # visualize path
    dbg = np.zeros((*area.shape[1:], 3))
    dbg[:,:,0] = area[0]
    dbg[:,:,1] = area[1]
    dbg[:,:,2] = self._moves[:, cx-dim:cx+dim, cy-dim:cy+dim]
    cv2.imshow('gm', cv2.resize(dbg, (945, 945)))
    ###

    return np.zeros((2, ))