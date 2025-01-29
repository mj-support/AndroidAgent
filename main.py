import gymnasium as gym
from stable_baselines3 import DQN
import environment

def train(env, task):
    model = DQN("MultiInputPolicy", env=env, verbose=1, learning_rate=0.001, gamma=0.99, exploration_fraction=0.3)
    model.learn(total_timesteps=2500)
    model.save(f"models/{task}.zip")

def predict(env, task):
    model_path = f"models/{task}.zip"
    model = DQN.load(model_path, env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    # Test model
    print(f"'{model_path}' Model loaded successfully. Start testing...")

    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

def main():
    # Create environment
    env_id = "Android-v0"
    emulator_id = "emulator-5554"   # Default emulator-name
    task = "airplane"
    env = gym.make(env_id, emulator_id=emulator_id, task=task)

    train(env, task)
    predict(env, task)

if __name__ == "__main__":
    main()
