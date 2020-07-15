# GlobalMap

This module uses a neural network for creating a global map of a location from a sequence of minimap states. You can read [more technical details](training) in `training` folder.

Estimated development plan:

- A simple algorithm for combining multiple intersected minimaps into a global map.

- A neural network that combines two intersected minimaps into a global map.

- A neural network that combines old global map and new state of minimap into a new global map.

- Creating a new of the global map from the old state and a screenshot of the minimap. (without MinimapRecognizer)