import cv2
class CBotDebugState:
  def __init__(self, screenshot, logger):
    self.logger = logger
    self._data = dict()
    self.put('screenshot', screenshot)
    pass
  
  def put(self, name, value):
    self._data[name] = value
    return
  
  def show(self, configs):
    if not configs: return
    
    self.logger.info('working')
    for name in configs:
      cv2.imshow(name, 
        cv2.resize(self._data[name], (640, 480))
      )
    pass