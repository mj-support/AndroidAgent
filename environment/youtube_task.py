import os


class YoutubeTask:
    """
    Represents a task for displaying a specific Youtube video.
    This class provides methods for resetting the task and evaluating the rewards based on the interactions with the UI.
    """
    def __init__(self, emulator_id, token="Charlie bit my finger! ORIGINAL", exploration_mode="full_exploration", episode_timesteps=100):
        """
        Initializes the AirplaneTask instance.

        Args:
            emulator_id (str):          The ID of the emulator.
            token (str):                The token of given to the agent to allow him to type words into text fields.
                                        Defaults to "Charlie bit my finger! ORIGINAL".
            exploration_mode (str):     Exploration mode of the agent, also affecting the reward structure.
                                        Possible modes: "guided_restricted", "guided_open", "full_exploration".
                                        Defaults to "full_exploration"
            episode_timesteps (int):    The maximum number of steps performed per episode. Defaults to 100.
        """
        self.emulator_id = emulator_id
        self.token = token.replace('\\', '')
        self.exploration_mode = exploration_mode
        self.episode_timesteps = episode_timesteps
        self.given_rewards = {}

    def reset_task(self):
        """
        Resets the task to its initial state.

        Clears the Youtube app and ensures airplane mode is disabled and Wi-Fi is enabled.
        """
        self.given_rewards = {"s1": False, "s2": False, "s3": False, "s4": False, "s5": False, "s6": False, "s7": False}
        os.system("adb -s {0} shell pm clear com.google.android.youtube".format(self.emulator_id))
        os.system("adb -s {0} shell settings put global airplane_mode_on 0".format(self.emulator_id))
        os.system("adb -s {0} shell svc wifi enable".format(self.emulator_id))

    def get_reward(self, obs_history, ui_options_current):
        """
        Evaluates the reward based on the action text.

        Args:
            obs_history (dict): Previous observation history.

        Returns:
            Tuple[int, bool]: reward, done
        """
        reward = 0
        done = False
        package = obs_history[-1]['package']
        action_text = obs_history[-1]['action_text']
        intermediate_steps_to_reach_goal = 7
        finish_reward = self.episode_timesteps / 2
        intermediate_reward = finish_reward / intermediate_steps_to_reach_goal

        # Evaluate the action based on the action
        if self.exploration_mode == "guided_restricted" or self.exploration_mode == "guided_open":
            if package == "nexuslauncher" and action_text == "swipe up":
                # First step: Swipe up from the home screen
                if not self.given_rewards["s1"]:
                    reward = intermediate_reward
                    self.given_rewards["s1"] = True
                else:
                    reward = 1
                # Avoid getting rewards for pointless swiping and only rewarding swiping in the home screen
                print("Yay first step made!")
            elif package == "nexuslauncher" and action_text == "YouTube":
                # Second step: Tap "YouTube"
                if not self.given_rewards["s2"]:
                    reward = intermediate_reward
                    self.given_rewards["s2"] = True
                else:
                    reward = 1
                print("Yay second step made!")
            elif (package == "permissioncontroller" and
                  (action_text == "Allow" or action_text == "Don’t allow")):
                # Third step: Tap "Network & internet"
                if not self.given_rewards["s3"]:
                    reward = intermediate_reward
                    self.given_rewards["s3"] = True
                else:
                    reward = 1
                print("Yay third step made!")
            elif package == "youtube" and action_text == "swipe up" and len(ui_options_current) <= 3:
                if not self.given_rewards["s4"]:
                    reward = intermediate_reward
                    self.given_rewards["s4"] = True
                else:
                    reward = 1
                print("Yay fourth step made!")
            elif package == "youtube" and action_text == "Accept all" or action_text == "Reject all":
                if not self.given_rewards["s5"]:
                    reward = intermediate_reward
                    self.given_rewards["s5"] = True
                else:
                    reward = 1
                print("Yay fifth step made!")
            elif package == "youtube" and action_text == "Search" or action_text == "Search YouTube":
                if not self.given_rewards["s6"]:
                    reward = intermediate_reward
                    self.given_rewards["s6"] = True
                else:
                    reward = 1
                print("Yay sixth step made!")
            elif package == "youtube" and action_text == "Text field Search YouTube":
                if not self.given_rewards["s7"]:
                    reward = intermediate_reward
                    self.given_rewards["s7"] = True
                else:
                    reward = 1
                print("Yay seventh step made!")

        reward -= 1

        if package == "youtube" and self.token in action_text and action_text != self.token:
            reward = finish_reward
            print("Yay FINISHED!!")
            done = True

        return reward, done
