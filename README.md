# Lunar Lander

In this project, I will train an AI spaceship to land in between two flags on the Moon.

## How will I do this?

I will use Deep Reinforcement Learning to train the AI. If it is on the right track, I will reward it, and if not, it will be punished. It will learn to make the decisions which will give it the most reward. 
I use MlpPolicy, which is a type of neural network structure for policies (policy is what defines the strategy/behavior of an agent in an envrionment). It takes in info about the current state/observation and learns what kind of action to take using a Deep Reinforcment Algorithm called PPO (Proximal Policy Optimization), which teaches the model to optimize the policy/strategy without changing the policy too much, as that could lead to unstable learning. 

# Simple Overview

## 1) Data Collection

The agent interacts with the environment to collect a batch of data consisting of states, rewards, actions, and other important factors about the state.

## 2) Compute Advantage

The algorithm then estimates the advantage of each action taken within each trajectory or state. Advantage is how better or worse the action performed was from what was expected.

## 3) Update Policy 

After we compute the advantage, we update our policy. We use a surrogate objective function to optimize the strategy to improve the actions taken by the agent.
  -> Surrogate Objective Function: 
    1) we measure the policy change between the old and new policies.
    2) We obtain the clipped ratio, which we get by limiting the ratio to be between (1 - clip_param) and (1 + clip_param) -> note: clip_param is a small positive number used to control the policy change
    3) We combine the ration and the clipped ratio into a function. 

## 4) Repeat

We need to repeat this process for thousands if not millions of epochs for the AI to learn to make significant changes which will lead it to achieve its goal.
  


