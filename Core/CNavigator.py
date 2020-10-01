from GlobalMap.CGlobalMap import CGlobalMap

class CNavigator:
  def __init__(self):
    self._globalMap = CGlobalMap()
    return
  
  def update(self, maskWalls, maskUnknown):
    self._globalMap.update(maskWalls, maskUnknown)
    pass
  
  def nearestGoal(self):
    goal = self._globalMap.process()
    return goal