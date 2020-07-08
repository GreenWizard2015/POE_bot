def rangeWithOverlap(a, b):
  for x in range(0, 1 + a - b, b):
    yield x

  last = a - b
  if 0 < (last % b):
    yield last
  return
      
def splitRegions(h, w, rh, rw):
  regions = []
  for x in rangeWithOverlap(w, rw):
    for y in rangeWithOverlap(h, rh):
      regions.append((x, y))

  return regions
