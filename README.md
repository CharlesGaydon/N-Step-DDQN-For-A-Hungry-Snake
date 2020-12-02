# snake-autoplay-RL
Learning via auto-play a strategy for two player snake game with reinforcement learning.

## Todo

- [X] add Blak to git projet : https://github.com/psf/black
- [X] Add requirements yaml file for conda env python 3.7 and numpy for now and black.
- [X] add env to pycharma nd add black directly in it https://github.com/psf/black/blob/master/docs/editor_integration.md
- [X] add github actions for black
- [X] create logic for the game:
  - [X] initialization of the board
  - [X] keep track of whose turn it is and who is playing at each time
- [X] Create a random player strategy
- [X] Create a pit.py script to have random player fight each other
- [X] Add evaluation of agents at each iteration alognside comparison of models
- [X] Create a NNEt which acts as an Agent
  - [ ] Add additional metadata to the agent input, which size is a parameter of the agent's model
  - [X] Create a model class that can be trained
  - [X] Check  checkpoints saving
  - [X] Test loading saved model in pit
- [ ] Add a conf folder with full training and test confs, and a global mode that can be activated as param of main and pit
- Model:
  - [ ] add early stopping
  - [ ] add decreasing probability to select the samples
  - [ ] add a parameter threshold to only use most recent examples
- [ ] Reinforcement Learning:
  - [ ] Create a NN based Agent, which gets the game and the NNetWrapper OR Modify the wrapper directly by adding decide method
  - [ ] Distinguish : the NNET which approximate state value using pi * states values VS the policymaker=agent which give final proba
- [X] Create a pit option to compare model agent with a random player
- [X] Create an auto-play scheme that gives (s, a, r) triplets (embed it in the Arena)
- [X] Create a coach that lauches a serie of autoplays and sequentially train the model, with a simple agent update rule.
- [X] Give credit to alpha zero implementation
- [ ] Add logging
- [ ] Add some documentation
  - Document functions
  - add sphinx and automated doc

## Credit

Many thanks to the team behind [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/), which greatly inspired this repository structure.
