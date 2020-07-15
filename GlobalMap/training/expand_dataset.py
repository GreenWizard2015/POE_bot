import os
import glob
import shutil

datasetFolder = os.path.join(os.path.dirname(__file__), 'dataset')
raw = os.path.join(datasetFolder, 'raw')

for p in glob.glob(os.path.join(raw, '*')):
  # copy /dataset/raw/**/*.* into /dataset/**/*.* 
  print('Processing /%s...' % os.path.basename(p))
  dest = os.path.join(datasetFolder, os.path.basename(p))
  shutil.rmtree(dest, ignore_errors=True)
  shutil.copytree(p, dest)
  
  # TODO: Use CMinimapRecognizer and create masks for each *_input.jpg
  print('Creating masks...')
  # TODO: Combine [i]_*.jpg and [i + 1]_*.jpg into [i]_[i + 1]_*.jpg
  print('Creating global masks...')
  print()