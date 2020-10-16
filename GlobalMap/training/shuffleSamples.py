import os
import glob
import time
import shutil
import random

src = os.path.abspath('../../minimap')
dest = os.path.abspath('dataset')

samples = glob.glob(os.path.join(src, '*_map.jpg'))
prefix = '%d' % time.time()
for i in range(len(samples)):
  sampleInd = random.randint(0, len(samples) - 1)
  s = samples[sampleInd]
  del samples[sampleInd]
   
  shutil.copy(s, os.path.join(dest, '%s_%d_map.jpg' % (prefix, i)))
  shutil.copy(s.replace('_map', '_target'), os.path.join(dest, '%s_%d_target.jpg' % (prefix, i)))