import cv2
class CBotDebugState:
  def __init__(self, screenshot, logger):
    self.logger = logger
    self._screenshot = screenshot
    pass
  
  def show(self, configs):
    if not configs: return
    
    self.logger.info('working')

    cv2.imshow('$Output window$', 
      cv2.resize(self._screenshot, (640, 480))
    )
    pass