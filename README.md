# Reinforcement-Learning-Learning
This repo is for projects I've done to learn and practice Reinforcement Learning

For dqn, I didn't update Q every action step as should be done when using TD(0). I also didn't update each new state action pair with a set batch. Doing both these things was quite slow in combination (especially fitting on every action step).

What I decided to do was treat the Q network more like a neural network. I collected all state action pairs for an entire episode and add these with a large batch (128) of previous pairs (experience replay). Then I fit this entire group for 10 epochs (normally only 1 epoch is used as Q is updated every action step). I found that this greatly increased my run time and greatly increased the stability and consistency of the network. Before it seemed hit or miss off the start whether the agent would converge on a solution. Now it seems to do it reliably (although keeping it at the top once it's reached it is another story).

One issue with this is that the action targets (what the NN is trying to fit too) don't change as the model changes (for the 10 epochs). Remember that these targets are only there because a single action was performed and yet the NN needs values for all the actions to fit too. While intuitively this sounds like it would greatly reduce the performance, perhaps this is part of the reason for the increased stability compared to other methods.
