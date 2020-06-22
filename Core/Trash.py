
def clarifyCountours(mask, iterations=7, incFactor=1, penaltyFactor=20, kernelN=2):
  for m in range(iterations):
    addFactor = sqrt(1 + m * incFactor)
    kernel = np.ones((kernelN, kernelN)) / ((kernelN ** 2) / addFactor)
    mask = cv2.filter2D(mask, -1, kernel.astype(np.float32))
    
    maskMin = min((200, m * penaltyFactor))
    mask = cv2.bitwise_and(mask, cv2.inRange(mask, maskMin, 255))

  return cv2.erode(mask, np.ones((3, 3)))
