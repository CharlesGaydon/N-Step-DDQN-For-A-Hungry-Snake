# snake-autoplay-RL
Learning via auto-play a strategy for two player snake game with reinforcement learning. 


- [X] add Blak to git projet
- [X] create logic for the game:
  - [X] initialization of the board
  - [X] keep track of whose turn it is and who is playing at each time
- [X] Create a random player strategy
- [X] Create a pit.py script to have random player fight each other
- [ ] Create a general Agent that can use any model with the board
  - [ ] Add additional metadata to the agent input, which size is a parameter of the agent's model
  - [ ] Create a model class that can be trained and saves checkpoints
- [ ] Create a pit option to compare model agent with a random player
- [ ] Create an auto-play scheme that gives (s, a, r) triplets (embed it in the Arena)
- [ ] Create a coach that lauches a serie of autoplays and sequentially train the model, with a simple agent update rule.
- [ ] Give credit to alpha zero implementation
