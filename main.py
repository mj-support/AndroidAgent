import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Box
import numpy as np
import os
import gym_android.envs
import torch

"""class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=np.concatenate([space.low.flatten() for space in env.observation_space.spaces.values()]),
            high=np.concatenate([space.high.flatten() for space in env.observation_space.spaces.values()]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.concatenate([value.flatten() for value in observation.values()])"""

def train(env, env_id):
    """env = Monitor(env, filename="./logs/monitor.csv") # info_keywords=("episode_reward",))

    # Callback to evaluate and save the best model
    eval_env = gym.make(env_id, max_episode_steps=25, render_mode="rgb_array")
    eval_env = FlattenDictWrapper(eval_env)
    eval_env = Monitor(eval_env, filename="./logs/eval_monitor.csv") #info_keywords=("episode_reward",))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=1,  # Evaluate every 25 steps
        deterministic=True,
        render=False,
    )

    # Create the PPO model with n_steps=25
    #model = PPO("MlpPolicy", env=env, n_steps=25, batch_size=1, verbose=1, n_epochs=4, ent_coef=0.01, normalize_advantage=False)
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], activation_fn=torch.nn.ReLU)
    model = PPO("MlpPolicy", env, learning_rate=1e-3, gamma=0.95, n_steps=512, batch_size=64, n_epochs=10, ent_coef=0.05, clip_range=0.3, policy_kwargs=policy_kwargs, verbose=1)

    # Train the model with callbacks
    model.learn(
        total_timesteps=100,
        callback=[eval_callback, checkpoint_callback],
    )"""

    # Separate evaluation env
    eval_env = gym.make(env_id, max_episode_steps=15)
    #eval_env = FlattenDictWrapper(eval_env)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/best_model/",
                                 log_path="./logs/eval/", eval_freq=1,
                                 deterministic=True, render=False)

    # create model
    ALGO = PPO
    model = ALGO("MlpPolicy", env=env, verbose=1)
    # train model
    model.learn(total_timesteps=1000, tb_log_name="first_run", callback=eval_callback)#, progress_bar=True)
    # Save the final model
    model.save("./logs/omg_first_gym_test")

def predict(env):
    model_path = "/Users/marco/repos/best_model.zip"
    model = PPO.load(model_path, env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    # Modell testen
    print(f"Modell '{model_path}' erfolgreich geladen. Starte Test...")

    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

        #vec_env.render("human")

        if done:
            obs = vec_env.reset()



def main():
    env_id = "Android-v0"
    env = gym.make(env_id, max_episode_steps=15)
    #env = FlattenDictWrapper(env)
    train(env, env_id)
    #predict(env)


if __name__ == "__main__":
    os.makedirs("./logs", exist_ok=True)
    main()
    """env = Monitor(env)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=25,  # Evaluate every 10,000 steps
        deterministic=True,
        render=False,
    )"""