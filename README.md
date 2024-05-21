# pytorch-a2c-ppo-acktr

## Fork

This is Andrew Liu's fork of pytorch-a2c-ppo-acktr-gail

Here I will be making some slight adjustments to code to make it nicer to track experiments with

### Changes from Original

In each of the files, you can search "Andy:" for places where the original code was changed.

* main.py: add Tensorboard and log every update
  * also add optionality to using wandb (stopped working on CHPC)
  * add optionality to save checkpoints
* a2c_ppo/arguments.py: add flags for tracking and video capture and checkpointing and using specific kwargs
* a2c_ppo/algo/ppo.py: add calculations for approx kl divergence and clipfracs
* a2c_ppo/envs.py: add video capture wrapper to environments, ability to use env_kwargs
* a2c_ppo/model.py: added FlexBase for changing how many layers are shared between actor and critic

* evaluate.py: adding code to return all seen obs and actions during evaluation, as well as hidden states so that we could potentitally map out hidden state trajectories. Adding code to take an optional callback to gather additional data from environment, and option to select how many episodes
  * added ability to use to record video and verbosity for printing individual episode results


## Code Flow Notes

To generate a model, main.py creates a 

Policy (model.py)
* Uses MLPBase if observation is 1-dim, CNNBase if 3-dim
* self.base = MLPBase: base is the network used in Policy
* base outputs a critic value, actor hidden activatios, and rnn hidden states
* self.dist converts actor hidden activations to action output. For discrete actions, it is Categorical (from distributions.py) which adds a hidden -> n linear output and does categorical distribution
* save rollouts during episode to self.rollouts, which is an instance of RolloutsStorage

MLPBase
* Two separate networks are created for actor and critic
* Tanh activations used
* actor and critic have layers input -> hidden -> hidden (2 hidden layers), default to 64 hidden units
* critic has built in final layer hidden -> 1, so outputs value
* if self.recurrent, adds in a GRU layer input -> hidden, and changes the inputs of actor and critic to hidden

RolloutsStorage
* For recurrent policy, saves subsequent hidden states detached, so no rollback through time during updates
