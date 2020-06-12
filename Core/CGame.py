from pywinauto import handleprops
import win32gui

class CGame:
  def __init__(self, logger):
    self.logger = logger
    self._lastHWND = None
    self._lastWndIsGame = False
    pass
  
  def isActive(self):
    hwnd = win32gui.GetForegroundWindow()
    if not self._lastHWND == hwnd:
      self._lastHWND = hwnd
      self._lastWndIsGame = 0 <= handleprops.text(hwnd).lower().find('path of exile')

    return self._lastWndIsGame
  
  def screenshot(self):
    # @todo: take game window screenshot
    return None
  
  def execute(self, actions):
    # @todo: send actions to game window
    return False