import cv2
import numpy as np

MAP_WALL = 255
MAP_UNDISCOVERED = 127

def mapMask(img):
  marker = [0, 0, 200]
  mask = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
  
  mask[np.where(mask[:,:,2] < 85)] = marker
  mask[np.where(mask[:,:,2] > 135)] = marker
  mask[np.where(mask[:,:,0] < 130)] = marker
  
  return (marker == mask[:, :]).all(axis=2)
  
def extractGameMap(img, mapHeight=256, mapWidth=256, indentRT=(15, 15), returnSource=False):
  _, w, _ = img.shape
  w -= indentRT[0] + mapWidth
  img = img[int(indentRT[1]):int(indentRT[1] + mapHeight), int(w):int(w + mapWidth)]
  src = img.copy() if returnSource else None
  
  mask = mapMask(img)

  img = img[:, :, 2] // 8
  img[np.where(mask)] = 0
  
  newArea = np.isin(img, np.arange(1, 12))
  
  img[np.where(newArea)] = MAP_WALL
  img[~(np.logical_or(newArea, mask))] = MAP_UNDISCOVERED

  return (img, src)