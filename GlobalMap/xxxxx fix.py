#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path
import shutil
import os
import re
from codecs import ignore_errors

pattern = re.compile(r's:(\d+):\\"(.*?)\\";')
escapedChars = re.compile(r'\\[0-9]{1,3}')

def lines(nm):
  with open(nm, 'r', encoding='utf8') as f:
    for line in f:
      yield line

def onMatch(m):
  oldLen, text = m.groups()
  text = text.lower()
  oldLen = int(oldLen)
  
  newLen = oldLen + \
    (text.count('casino') * (9 - 6)) + \
    (text.count('game') * 7)
     
  if not (newLen == oldLen):
    print(m.group(0))
    return 's:%d:\\"%s\\";' % (newLen, text)
  return m.group(0)

def fix(data):
  res = pattern.sub(onMatch, data) \
    .replace('casino', 'bookmaker') \
    .replace('Casino', 'Bookmaker') \
    .replace('Game', 'Application') \
    .replace('game', 'application')
  
  return res

inputFile = 'd:/23/database-betrush-ru-1602770680.sql'
dest = 'd:/db.sqL'

with open(dest, 'w', encoding='utf8') as f:
  for l in lines(inputFile):
    f.write(fix(l))