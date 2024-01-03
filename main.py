import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the Lunar Lander environment
env = DummyVecEnv([lambda: gym.make('LunarLander-v2', render_mode="human")])
print("[*] Env Made")

model = PPO.load("ppo_lunar_lander")

print("[*] Model Loaded")

# Evaluate the trained agent
total_rewards = 0
num_eval_episodes = 10

for _ in range(num_eval_episodes):
    state = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        print(f'Reward: {reward}')
        total_rewards += reward

average_reward = total_rewards / num_eval_episodes
print(f"Average Reward over {num_eval_episodes} episodes: {average_reward}")

# Close the environment
env.close()
