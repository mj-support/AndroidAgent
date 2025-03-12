import gymnasium as gym
from stable_baselines3 import DQN, PPO
import environment

def train(env, task):
    model = DQN("MultiInputPolicy", env=env, verbose=1, learning_rate=0.001, gamma=0.99, exploration_fraction=0.5, seed=42)
    #model = PPO("MultiInputPolicy", env=env, verbose=1, ent_coef=0.1, learning_rate=3e-4, n_steps=20, batch_size=20)
    model.learn(total_timesteps=7500)
    model.save(f"models/{task}.zip")
    print(f"Finished training - model saved in models/{task}.zip\n")

def predict(env, task):
    model_path = f"models/{task}.zip"
    model = DQN.load(model_path, env=env)
    model.learn(total_timesteps=1000, reset_num_timesteps=False)

    vec_env = model.get_env()
    obs = vec_env.reset()

    # Test model
    print(f"'{model_path}' Model loaded successfully - start testing...")

    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

def main():
    # Create environment
    env_id = "Android-v0"
    emulator_id = "emulator-5554"   # Default emulator-name
    task = "airplane" #"youtube" #"airplane"
    env = gym.make(env_id, emulator_id=emulator_id, task=task)

    train(env, task)
    predict(env, task)

if __name__ == "__main__":
    main()
