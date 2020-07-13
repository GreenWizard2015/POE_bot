# POE_bot

**ATTENTION!** This project was created and developed solely for the purpose of studying AI / ML.

[Path Of Exile](https://www.pathofexile.com/) is a relatively simple (in terms of automation), multifaceted and dynamic game. It has a fairly wide range of tasks that can be solved gradually and in various ways. For example, a few years ago I mastered the basics of writing an extremely [primitive bot](https://gist.github.com/GreenWizard2015/6fd90e0b49eda1354ad549b66397e946) ([video](https://www.youtube.com/watch?v=PELXt_utwu4)) that could go around one specific map using only simple image processing.

The ultimate goal of the project can be formulated as follows:

> The bot should be able to explore any solid map (without teleports, additional zones, ignoring quests and event triggers), avoid enemies attacks and respond to them.
> 
> The bot should use only the image of the game process, without hacking the game, its resources, and using the previously collected information about the cards.

Estimated development plan:

- The basic structure of the bot, capturing the image of the game.

- Reading the current state of the minimap. ([MinimapRecognizer](https://github.com/GreenWizard2015/POE_bot/tree/master/MinimapRecognizer))

- Combining multiple minimap states into one global map.

- Building a step-by-step strategy for exploring the global map.

- Recognition of enemies and their attacks.
