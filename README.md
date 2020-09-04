# Environment_2048

# Description
  C++ implementation of Gabrielle Cirulli's 2048 game, designed for Reinforcement Learning experiments in Pytorch while also using Cython for communication.

# Requirements
- Visual Studio 2019 or any compatible C++ compiler
- Anaconda
- The following python libraries:
	- Cython: 0.29.14 (or higher)
	- Matplotlib: 3.1.1 (or higher)
	- Numpy: 1.18.1
	- Pandas: 0.25.3
	- Pytorch: 1.4.0

# Currently implemented algorithms:
	- Reinforce
	
# Sometime in the future:
	- Deep Q Network
	- Double Deep Q Network
	- Duelling Double Deep Q Network
	
# Additional requirements:
- In order to run the unit tests, it's mandatory to have the [Catch Adapter for Visual Studio](https://marketplace.visualstudio.com/items?itemName=JohnnyHendriks.ext01)
- Instructions to set up the unit tests: [DavidZi's Answer](https://stackoverflow.com/questions/59645381/best-practices-for-unit-testing-with-catch2-in-visual-studio)

# More details:
  The project contains an environment where you can implement, test and benchmark different algorithms on 2048. There are some helper classes
  for logging and plotting and output data in the **LOGS** folder. Current collected information include:
	- Episode length
	- Total episode reward
	- State, action and reward at each step for each episode
	- Entropy
	- Network output
	- Minimum and maximum reward
	- Max cells obtained
  To further increase the speed of the agent, the logging is semi-instant. Meaning that for an episode the log for state, action and reward will be written when
  that episode ends.
  The parameters can be changed in the `Parameters` class. The currently supported arguments are:
	- Gamma
	- Learning Rate
	- Episodes
	- Board size
	- Input size
	- Output size

# References
- The algorithm is taken from the book [Foundations of Deep Reinforcement Learning: Theory and Practice in Python](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Hands/dp/0135172381)
	