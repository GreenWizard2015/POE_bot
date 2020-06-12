#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
from Core.CBot import CBot
from Core.CGame import CGame

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
  debugOutput = True
  bot = CBot(logger)
  game = CGame(logger)
  while bot.isActive():
    time.sleep(0)
    if not game.isActive(): continue
    
    actions, debugInfo = bot.process(game.screenshot())
    game.execute(actions)
    debugInfo.show(debugOutput)
    pass

if __name__ == '__main__':
  main()
