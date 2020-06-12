class CBotDebugState:
  def __init__(self, screenshot, logger):
    self.logger = logger
    self._screenshot = screenshot
    pass
  
  def show(self, configs):
    if not configs: return
    
    self.logger.info('working')
    # @todo: show screenshot
    pass