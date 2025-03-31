import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import environment
from eval import MetricsCallback, create_log_dir


def train(env, task, total_timesteps=1000, episode_timesteps=100):
    """Train a reinforcement learning model using DQN.

    Args:
        env: The environment in which the agent will be trained.
        task (str): A string identifier for the training task.
        total_timesteps (int): Total number of timesteps for training. Default is 1000.
        episode_timesteps (int): Number of timesteps per episode for evaluation. Default is 100.

    Returns:
        tuple: A tuple containing the trained model and the log directory path.
    """
    # Create log directory
    log_dir = create_log_dir(task)

    # Wrap the environment with Monitor
    env = Monitor(env, log_dir)

    # Create the model
    model = DQN("MultiInputPolicy", env=env, verbose=1, learning_rate=0.001, gamma=0.99, exploration_fraction=0.5, seed=42, tensorboard_log=log_dir)

    # Create a metrics callback
    metrics_callback = MetricsCallback(log_dir=log_dir, eval_freq=episode_timesteps, success_threshold=episode_timesteps/2.0)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)

    # Save the model
    model_path = f"{log_dir}/{task}.zip"
    model.save(model_path)
    print(f"Finished training - model saved in {model_path}")

    # Generate and save final training metrics visualization
    metrics_callback.plot_metrics()
    print(f"Training metrics saved in {log_dir}")

    return model, log_dir


def predict(env, task, total_timesteps=1000, log_dir=None):
    """Load a trained DQN model and perform evaluation.
    
    Args:
        env: The environment in which the agent will be tested.
        task (str): A string identifier for the task.
        total_timesteps (int): Number of timesteps for evaluation. Default is 1000.
        log_dir (str): Directory where the trained model is stored. Default is None.
    """
    print("\nStarting prediction...\n")
    model_path = f"{log_dir}/{task}.zip"
    model = DQN.load(model_path, env=env)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    print(f"'{model_path}' Model loaded successfully - start testing...")

    vec_env = model.get_env()
    obs = vec_env.reset()

    for _ in range(total_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)


def main():
    # Create environment
    env_id = "Android-v0"
    emulator_id = "emulator-5554"  # Default emulator-name
    # Task options: "youtube", "airplane"
    task = "airplane"
    # Mode options: "guided_restricted", "guided_open", "full_exploration"
    exploration_mode = "guided_restricted"
    episode_timesteps = 100
    total_timesteps = 1000
    env = gym.make(env_id, emulator_id=emulator_id, task=task, exploration_mode=exploration_mode, episode_timesteps=episode_timesteps)

    # Train the model
    model, train_log_dir = train(env, task, total_timesteps=total_timesteps, episode_timesteps=episode_timesteps)

    # Continue training and evaluate
    predict(env, task, total_timesteps=total_timesteps, log_dir=train_log_dir)


if __name__ == "__main__":
    main()
