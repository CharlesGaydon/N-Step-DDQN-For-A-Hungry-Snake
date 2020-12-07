# snake-autoplay-RL
Learning via auto-play a strategy for two player snake game with reinforcement learning.

## Todo


- [X] Add a mask to avoid refitting on previous predictions in optimize_network
- [X] implement a n step q-learning
- [X] Remove "not useful experiences" with a probability depending on the reward.
- [ ] Stop episodes stuck in a loop sooner (e.g. 50 steps)
- [ ] Add logging
- [ ] Add some documentation
- [X] read https://towardsdatascience.com/why-going-from-implementing-q-learning-to-deep-q-learning-can-be-difficult-36e7ea1648af
  - Document functions
  - add sphinx and automated doc


# Lessons learnt
- Use standard package like `gym` to have a standard implementation of an environment
- Start by a simpler problem and reuse later the architecture
- Double Deep Q-Network theory and practice: using a target network and experience replay
-


## Credit

Many thanks to the team behind [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/), which greatly inspired this repository structure.
