# snake-autoplay-RL

---

Using a **n-step Double Deep Q-Network (DDQN) with Experience Replay**, a snake learns how to eat apples by trial and errors,

---

**NOTE**:
Running this code might work, but full from-scratch environment creation
 and scripts running was not tested.

 You could be inspired by
 the n-step deep learning implementation in arena.py,
 or by how masking is performed in DDQN, but generally speaking there are probably better
 implementation for the snake environment our there.

---

## Setting
- Small world (7x7)
- Reward: 10 for apple, -20 for crash, -2 * distance to apple at each step
- 3-step Double Deep Q-Network
- Probabilistic choice of experience to add to memory
- Parameters are in `main_1p.py`

---

## Results & Further step to bridge the gap with SOTA solutions

- `main_1p.py`
  - After 10,000 fittings steps: mean loss = 0.019
- `demonstrate_1p.py`
  - Evaluation on 30 games:
    - Median episode duration: 75.0
    - Median reward: 3.0
    - High score: 13
- `demonstrate_1p.py --mode game`
  - Example game board with a snake that ate 5 apples:
![result](./Trained_Models/img/snake_ascii_art.png)


NB: games were cut at 75 if reward was below 2.

This project started as a two-player snake arena and was then upgraded with the sole goal of getting a better sense of challenges and constraints of
faced when implementing reinforcement learning algorithms from scratch.

Now that I got a good hang of all the concepts, being thorough in experimenting and optimizing would require to
refactor all the code from scratch to really control for the designs, which seems tedious and time-consuming considering the
recreational nature of this project.

**As a results, I am not considering a systematic parameter optimization, and will not try to optimize further all metaparameters.**

Authors in [Finnson & Morlo](https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf)
shows shows that average scores of ~30 apples could be achieved (although in a space 3 times the size of our ownn)
Simpler solutions include using custom features as in this [article (2019)](https://towardsdatascience.com/why-going-from-implementing-q-learning-to-deep-q-learning-can-be-difficult-36e7ea1648af)
from Ray Heberer, although limited in potential.



## Credit
- Sutton and Barto's [Reinformcent Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)
for pseudo-code of n-step SARSA logic.
- The [article (2019)](https://towardsdatascience.com/why-going-from-implementing-q-learning-to-deep-q-learning-can-be-difficult-36e7ea1648af)
from Ray Heberer for ideas related to Double Deep Q-Network.
- This [article (2020)](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)
from Hennie de Harderr for the idea of rewarding getting closer to the apple.
- The team behind [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/),
which greatly inspired this repository structure and classes, at least at the beginning.

---

## Good ideas and lessons for next time
Reinforcement LearningL:
- Go beyond Deep Q-Network with Double Deep Q-Network and Experience Replay.
- Go beyond 1-step Q-Network with n-step Q-Network, but be aware of the resulting complexity.
- Use standard reinforcement learning libraries like `gym`; it makes a more modular implementation.

Code
- Use `black` for assuring PEP-8 convention compliance saves a huge amount of time.
- Start first by considering the existing literature to get a sense of feasibility e.g. [Finnson & Morlo](https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf)
- Start by a simpler problem : 1-player before the harder 2-player snake game; cartpole before snake game.
- Save model's weights frequently along learning, to be able to resume learning in another session.
