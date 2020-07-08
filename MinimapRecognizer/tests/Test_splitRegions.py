from MinimapRecognizer.splitRegions import splitRegions

class Test_splitRegions:
  def test_exactMatch(self):
    res = splitRegions(2, 2, 2, 2)
    assert str(res) == '[(0, 0)]'
    
  def test_exactSplit(self):
    res = splitRegions(4, 4, 2, 2)
    assert str(res) == '[(0, 0), (0, 2), (2, 0), (2, 2)]'
    
  def test_overlap(self):
    res = splitRegions(4, 4, 3, 3)
    assert str(res) == '[(0, 0), (0, 1), (1, 0), (1, 1)]'