import random
import numpy as np
import cv2

def largest_indices(ary, n):
  flat = ary.flatten()
  indices = np.argpartition(flat, -n)[-n:]
  indices = indices[np.argsort(-flat[indices])]
  return np.unravel_index(indices, ary.shape)

def mostFrequentElement(a):
  (values, counts) = np.unique(a, return_counts=True)
  ind = np.argmax(counts)
  return values[ind] 

def findGlobalShift(A, B, sz=32, minProbePoints=10, N=1000, minMatchCoef=0.9):
  vectors = []
  while len(vectors) < N:
    probePosX = random.randint(0, A.shape[0] - sz)
    probePosY = random.randint(0, A.shape[1] - sz)
    
    probe = A[probePosX:probePosX+sz, probePosY:probePosY+sz]
    if np.count_nonzero(probe) < minProbePoints: continue
    
    match = cv2.matchTemplate(B, probe, cv2.TM_CCORR_NORMED)
    matches = largest_indices(match, 5)
    
    for matchX, matchY in zip(matches[0], matches[1]):
      if match[matchX, matchY] < minMatchCoef: continue
      shiftX = probePosX - matchX
      shiftY = probePosY - matchY  
      vectors.append((shiftX, shiftY))
      
  vectors = np.array(vectors)
  return (
    mostFrequentElement(vectors[:, 0]),
    mostFrequentElement(vectors[:, 1])
  )