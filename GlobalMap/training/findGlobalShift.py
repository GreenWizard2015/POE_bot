import numpy as np
import cv2

def largest_indices(ary, n):
  flat = ary.flatten()
  indices = np.argpartition(flat, -n)[-n:]
  indices = indices[np.argsort(-flat[indices])]
  return np.unravel_index(indices, ary.shape)

def findGlobalShift(A, B):
  B = cv2.copyMakeBorder(B, *A.shape[:2], *A.shape[:2], borderType=cv2.BORDER_CONSTANT, value=0)
  
  match = cv2.matchTemplate(B, A, cv2.TM_CCORR_NORMED)
  
  ([ptsX], [ptsY])  = largest_indices(match, 1)
  anchor = (np.array(B.shape[:2]) - np.array(A.shape[:2])) // 2
  pts = anchor - np.array([ptsX, ptsY]) 
  return tuple(pts)