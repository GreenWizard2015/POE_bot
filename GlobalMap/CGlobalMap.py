import numpy as np
import cv2
import math
import skimage.measure
from GlobalMap.findPath import findPath

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
    self._mapArea = [math.inf, math.inf, -math.inf, -math.inf]
    self._pos = None
    self._realPos = np.array(self._map.shape[1:]) * 2 # Half of x4
    self.FORGET_RATE = 0.5
    self._changed = True
    self._targetPos = None
    return
  
  def update(self, maskWalls, maskUnknown):
    shift = np.zeros(2)
    self._changed = True
    if not (self._prev is None):
      shift = findShift(self._prev, maskWalls)
      self._changed = 25 < np.sum(np.power(shift, 2)) 
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
    # ignore central area
    c = dim // 2
    sz = 4
    masks[:, c-sz:c+sz+1, c-sz:c+sz+1] = oldArea[:, c-sz:c+sz+1, c-sz:c+sz+1]
    ###########
    self._map[:, x:x+dim, y:y+dim] = (oldArea + self.FORGET_RATE * masks) / (1 + self.FORGET_RATE)
    
    self._pos = (np.array([x, y]) + np.array([dim, dim]) / 2).astype(np.int32)
    self._prev = maskWalls
    return
  
  def process(self):
    if self._changed:
      cx, cy = self._pos
      area = np.where(100 < self._map, 255, 0)
      area[:, cx-1:cx+2, cy-1:cy+2] = 0 # cutoff center
      
      path, _ = findPath(area, pos=(cx, cy))
      self._targetPos = np.array(path[-2]) if path else None
      # visualize path
      dbg = np.zeros((*area.shape[1:], 3))
      dbg[:,:,0] = area[0]
      dbg[:,:,1] = area[1] 
      if path:
        for A, B in zip(path[1:], path[:-1]):
          cv2.line(dbg, tuple(A[::-1]), tuple(B[::-1]), (255, 255, 255))
      cv2.imshow('gm', cv2.resize(dbg[cx-25:cx+25, cy-25:cy+25], (945, 945)))
      ###

    if self._targetPos is None: return None
    return self._targetPos - self._pos