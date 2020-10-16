import glob
from os import path
import shutil
import os

def lines(nm):
  with open(nm, 'r', encoding='utf8') as f:
    for line in f:
      yield line

def transform(old):
  new = old \
    .replace('casino', 'bookmaker') \
    .replace('Casino', 'Bookmaker') \
    .replace('Game', 'Application') \
    .replace('game', 'application')
  
  return new, not (old == new)

folders = [
  'd://VirtualShare//Projects//timur//betrush//plugins',
  'd://VirtualShare//Projects//timur//betrush//themes//'
]

files = set()
for x in folders:
  for y in glob.iglob(path.join(x, '**/*.*'), recursive=True):
#     if y.endswith('.php') or y.endswith('.css') or y.endswith('.sql') or y.endswith('.pot'):
    if y.endswith('.sql'):
      files.add(y)

# replace content
# candidates = set() 
# for fname in files:
#   for l in lines(fname):
#     _, changed = transform(l)
#     if changed:
#       candidates.add(fname)

for fname in files:
  new = [transform(old)[0] for old in lines(fname)]
  with open(fname, 'w', encoding='utf8') as f:
    for l in new:
      f.write(l)
# renaming
for old in files:
  new, changed = transform(old)
  if changed:
    print(old)
    os.makedirs(os.path.dirname(new), exist_ok=True)
    shutil.move(old, new)
