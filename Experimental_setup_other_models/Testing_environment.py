# Import your environment
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # Ensure this file exists and contains your env class

env = SimpleKettleEnv()
done = False
output = env.step([3000])
total_reward = 0.0
while done == False:
    observations, reward, done,_, info = output
    # if observations[0] > 100:
    #     output = env.step(0)
    # else:
    #     output = env.step(1)
    output = env.step([3000])
    total_reward += reward
    print(f"Observation: {observations}") 
    print(f" Reward: {reward}, Done: {done}, Info: {info}, Total Reward: {total_reward}")
    print("------------------------------------")