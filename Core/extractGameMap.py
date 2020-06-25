def extractGameMap(img, mapHeight=256, mapWidth=256, indentRT=(15, 15)):
  _, w, _ = img.shape
  w -= indentRT[0] + mapWidth
  img = img[int(indentRT[1]):int(indentRT[1] + mapHeight), int(w):int(w + mapWidth)]
  return img.copy()