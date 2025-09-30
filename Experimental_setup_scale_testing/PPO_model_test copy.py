# ppo_kettle_evaluate.py

import gymnasium as gym
from stable_baselines3 import PPO
import time

# Import your custom environment
from Kettle_environment_updated_reward_fun import SimpleKettleEnv

# === 1. Configuration ===
# IMPORTANT: Update this path to point to your saved model file.
# The .zip extension is added automatically by stable-baselines3.
MODEL_PATH = "ppo_kettle_model/ppo_kettle_model_2000000_Kettle_environment_updated_reward_fun.zip"

# === 2. Instantiate Environment ===
# We use the same environment the model was trained on.
env = SimpleKettleEnv()

# === 3. Load the Trained Model ===
# The PPO.load() method reconstructs the model from the saved file.
print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded successfully. 🚀")

# === 4. Run a Single Episode ===
print("\n--- Starting Evaluation Episode ---")
obs, info = env.reset()
done = False
total_reward = 0.0
step_count = 0

while not done:
    # Get the action from the trained policy.
    # deterministic=True makes the agent choose the best action, not a random sample.
    action, _states = model.predict(obs, deterministic=False)

    # Perform the action in the environment
    obs, reward, done, truncated, info = env.step(action)

    # Update counters and print step info
    total_reward += reward
    step_count += 1
    
    # Use the render method to print a clean summary
    env.render()
    
    # Optional: Add a small delay to make it easier to watch in real-time
    # time.sleep(0.05)

# === 5. Print Final Results ===
print("\n--- Episode Finished ---")
print(f"Total Steps: {step_count}")
print(f"Final Total Reward: {total_reward:.4f}")
print("--------------------------\n")

# Clean up the environment
env.close()