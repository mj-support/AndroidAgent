import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os


class MetricsCallback(BaseCallback):
    """
    Custom callback for collecting and plotting training metrics.

    Tracks rewards, episode lengths, mean rewards, success rate, and other metrics
    during reinforcement learning training. Saves metrics periodically and generates plots.
    """

    def __init__(self, log_dir, eval_freq=500, verbose=1, success_threshold=None):
        """
        Initialize the callback.

        Args:
            log_dir (str): Directory to store logs and plots.
            eval_freq (int): Frequency (in timesteps) to save and plot metrics. Defaults to 500.
            verbose (int): Verbosity level (0 = silent, 1 = info messages). Defaults to 1.
            success_threshold: Threshold to consider an episode successful. Defaults to None.
        """
        super(MetricsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.success_threshold = success_threshold  # Threshold to consider an episode successful
        self.metrics = {
            'timesteps': [],
            'rewards': [],
            'episode_lengths': [],
            'mean_reward': [],  # Mean reward across all episodes
            'mean_episode_length': [],  # Mean episode length across all episodes
            'success_rate_percent': []  # Success rate as percentage
        }
        self.all_episode_rewards = []  # Store all episode rewards
        self.all_episode_lengths = []  # Store all episode lengths
        self.successful_episodes = 0  # Count of successful episodes
        self.total_episodes = 0  # Count of total episodes
        os.makedirs(log_dir, exist_ok=True)

        # Track the episode reward accumulation
        self.current_episode_reward = 0.0
        self.episode_started = False

    def _on_step(self) -> bool:
        """
        This method is called at every step of training.
        Tracks rewards, episode lengths, and computes success rates.
        Saves and plots metrics at specified intervals.

        Returns:
            True: Whether training should continue (always True in this implementation).
        """
        # Handle episode rewards tracking correctly
        info = self.locals.get('infos')[0]
        reward = self.locals.get('rewards')[0]
        done = self.locals.get('dones')[0]

        # Start tracking episode rewards on first step or after episode completion
        if not self.episode_started:
            self.current_episode_reward = 0.0
            self.episode_started = True

        # Accumulate reward for the current episode
        if reward is not None:
            self.current_episode_reward += reward

        # Process episode metrics when an episode is done
        if done:
            ep_len = None

            # Try to get episode length from different possible sources
            if 'episode' in info:
                # Stable Baselines VecMonitor format
                if 'l' in info['episode']:
                    ep_len = info['episode']['l']
                elif 'length' in info['episode']:
                    ep_len = info['episode']['length']

                # Some environments provide episode return directly
                if 'r' in info['episode']:
                    # Override with the monitor's reward if available (more accurate)
                    self.current_episode_reward = info['episode']['r']

            # If length not in info, try other methods
            if ep_len is None:
                ep_len = self.locals.get('episode_lengths')[0] if 'episode_lengths' in self.locals else None
                if ep_len is None and hasattr(self.training_env, 'get_episode_lengths'):
                    ep_len = self.training_env.get_episode_lengths()[-1]
                elif ep_len is None and hasattr(self.locals.get('env'), 'episode_length'):
                    ep_len = self.locals.get('env').episode_length

            # Record episode metrics
            self.metrics['timesteps'].append(self.num_timesteps)
            self.metrics['rewards'].append(self.current_episode_reward)
            self.all_episode_rewards.append(self.current_episode_reward)
            self.total_episodes += 1

            # Calculate and store mean reward across ALL episodes
            current_mean_reward = np.mean(self.all_episode_rewards)
            self.metrics['mean_reward'].append(current_mean_reward)

            # Track best mean reward
            if current_mean_reward > self.best_mean_reward:
                self.best_mean_reward = current_mean_reward
                if self.verbose > 0:
                    print(
                        f"New best mean reward: {self.best_mean_reward:.2f} over {len(self.all_episode_rewards)} episodes")
                    print(f"All episode rewards: {self.all_episode_rewards}")

            # Track success rate if threshold is provided
            if reward == self.success_threshold:
                self.successful_episodes += 1

            # Calculate success rate as percentage
            success_rate = (self.successful_episodes / self.total_episodes) * 100
            self.metrics['success_rate_percent'].append(success_rate)

            if ep_len is not None:
                self.metrics['episode_lengths'].append(ep_len)
                self.all_episode_lengths.append(ep_len)

                # Calculate and store mean episode length
                mean_episode_length = np.mean(self.all_episode_lengths)
                self.metrics['mean_episode_length'].append(mean_episode_length)

            # Reset episode tracking
            self.episode_started = False
            self.current_episode_reward = 0.0

        # Save metrics periodically
        if self.num_timesteps % self.eval_freq == 0:
            self.save_metrics()
            # Plot latest metrics
            if len(self.metrics['timesteps']) > 1:
                self.plot_metrics()

        return True

    def save_metrics(self):
        """
        Save collected metrics to a CSV file.
        """
        metrics_df = pd.DataFrame()

        # Add all collected metrics to dataframe
        for key, values in self.metrics.items():
            if len(values) > 0:
                # Pad shorter series with NaN
                if len(metrics_df) == 0:
                    metrics_df[key] = values
                else:
                    # Extend shorter series with NaN values
                    padded_values = values + [np.nan] * (len(metrics_df) - len(values))
                    metrics_df[key] = padded_values[:len(metrics_df)]

        # Save to CSV
        metrics_df.to_csv(os.path.join(self.log_dir, 'metrics.csv'), index=False)

    def plot_metrics(self):
        """
        Generate and save plots for the collected training metrics.
        """
        plt.figure(figsize=(15, 15))  # Made taller for more plots

        # Plot rewards
        plt.subplot(3, 2, 1)
        plt.plot(self.metrics['timesteps'], self.metrics['rewards'])
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Rewards')
        plt.title('Rewards per Episode')

        # Plot mean reward
        plt.subplot(3, 2, 2)
        plt.plot(self.metrics['timesteps'][:len(self.metrics['mean_reward'])],
                 self.metrics['mean_reward'], 'r-')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.title('Mean Episode Reward')

        # Plot episode lengths if available
        if len(self.metrics['episode_lengths']) > 0:
            plt.subplot(3, 2, 3)
            plt.plot(self.metrics['timesteps'][:len(self.metrics['episode_lengths'])],
                     self.metrics['episode_lengths'])
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Length')
            plt.title('Episode Length over Time')

        # Plot mean episode length
        if len(self.metrics['mean_episode_length']) > 0:
            plt.subplot(3, 2, 4)
            plt.plot(self.metrics['timesteps'][:len(self.metrics['mean_episode_length'])],
                     self.metrics['mean_episode_length'], 'g-')
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Episode Length')
            plt.title('Mean Episode Length')

        # Plot success rate if available
        if len(self.metrics['success_rate_percent']) > 0:
            plt.subplot(3, 2, 5)
            plt.plot(self.metrics['timesteps'][:len(self.metrics['success_rate_percent'])],
                     self.metrics['success_rate_percent'], 'y-')
            plt.xlabel('Timesteps')
            plt.ylabel('Success Rate (%)')
            plt.title('Episode Success Rate')
            plt.ylim(0, 100)  # Percentage range

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'metrics_{self.num_timesteps}.png'))
        plt.close()


def create_log_dir(task):
    """
    Create a timestamped log directory for storing training logs.
    
    Args:
        task (str): Name of the task for logging.
    
    Returns: 
        log_dir (str): Path to the created log directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"{task}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
