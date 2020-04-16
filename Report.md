## Learning Algorithm

The agent is based on Deep Q-Networks with simple (random) replay memory. 
The agent minimizes the MSE loss computed between expected and actual rewards using [Adam](https://arxiv.org/abs/1412.6980).
Other hyperparameters are given below:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

The agent's Q-Network is relatively simple, composed of two hidden layers with ReLU activation. 
The first layer contains 32 neurons and the second layer contains 8 neurons. 

## Plot of Rewards
![alt text](./train.png "Rewards per episode - agent is able to receive an average reward (over 100 episodes) of at least +13. ")  
The environment was solved in 437 episodes.

## Ideas for Future Work

This project uses Deep Q-Networks and it is able to solve the environment in less than 500 episodes. The performance can likely be 
improved with well know extensions to the Deep Q-Networks (DQN) algorithm, such as:
* [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461)
* [Prioritized experience replay](https://arxiv.org/abs/1511.05952)
* [Dueling DQN](https://arxiv.org/abs/1511.06581)  
