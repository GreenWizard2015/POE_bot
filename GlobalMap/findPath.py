import numpy as np
import random

DIRECTIONS = [
  [(1, 0), (0, 1), (-1, 0), (0, -1)],
  [(1, 1), (-1, 1), (1, -1), (-1, -1)]
]

def raycast(A, pos, direction):
  x, y = pos
  dx, dy = direction
  N = 0
  try:
    while (A[0, x, y] < 100) and (A[1, x, y] < 100):
      x += dx
      y += dy
      N += 1
  except:
    return (None, -1)
  return (np.array([x - dx, y - dy]), N)

def findPath(A, pos, maxQueue=200, minDist=3, minHitDist=1000):
  queue = [(pos, 0, [pos], 1)]
  hit = None
  hitDist = minHitDist
  while 0 < len(queue):
    startPt, startDist, path, lvl = queue.pop(0)
    if hitDist < startDist: continue

    dirs = DIRECTIONS[lvl % 2] if 2 < lvl else DIRECTIONS[0] + DIRECTIONS[1]
    for direction in dirs:
      pt, dist = raycast(A, startPt, direction)
      if dist < 1: continue
      
      x, y = pt + direction
      tdist = startDist + dist
      hitTarget = 0 < A[1, x, y]
      if hitTarget and (tdist < hitDist):
        hitDist = tdist
        hit = [pt] + path
        continue
      
      if (minDist <= dist) and (tdist < hitDist):
        queue.append((pt, tdist, [pt] + path, 1 + lvl))
        if maxQueue * 1.2 < len(queue):
          queue = list(sorted(
            queue, key=lambda x: x[1], reverse=True
          ))[:maxQueue]
  return(hit, hitDist)
