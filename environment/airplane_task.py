import os


class AirplaneTask:
    """
    Represents a task for enabling airplane mode on an Android emulator.
    This class provides methods for resetting the task and evaluating the rewards based on the interactions with the UI.
    """
    def __init__(self, emulator_id, token=None, exploration_mode="full_exploration", episode_timesteps=100):
        """
        Initializes the AirplaneTask instance.

        Args:
            emulator_id (str):          The ID of the emulator.
            token (str):                The token of given to the agent to allow him to type words into text fields.
            exploration_mode (str):     Exploration mode of the agent, also affecting the reward structure.
                                        Possible modes: "guided_restricted", "guided_open", "full_exploration".
                                        Defaults to "full_exploration"
            episode_timesteps (int):    The maximum number of steps performed per episode. Defaults to 100.
        """
        self.emulator_id = emulator_id
        self.token = token
        self.exploration_mode = exploration_mode
        self.episode_timesteps = episode_timesteps
        self.given_rewards = {}

    def reset_task(self):
        """
        Resets the task to its initial state.

        Clears the system settings app and ensures airplane mode is disabled and Wi-Fi is enabled.
        """
        self.given_rewards = {"n1": False, "n2": False, "n3": False, "sc1": False, "sc2": False}
        os.system("adb -s {0} shell pm clear com.android.settings".format(self.emulator_id))
        os.system("adb -s {0} shell settings put global airplane_mode_on 0".format(self.emulator_id))
        os.system("adb -s {0} shell svc wifi enable".format(self.emulator_id))

    def get_reward(self, obs_history):
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
        previous_action = ""
        intermediate_short_cut_steps_to_reach_goal = 2
        intermediate_normal_steps_to_reach_goal = 3
        finish_reward = self.episode_timesteps / 2
        intermediate_short_cut_reward = finish_reward / intermediate_short_cut_steps_to_reach_goal
        intermediate_normal_reward = finish_reward / intermediate_normal_steps_to_reach_goal

        if len(obs_history) > 1:
            previous_action = obs_history[-2]['action_text']

        if self.exploration_mode == "guided_restricted" or self.exploration_mode == "guided_open":
            # Evaluate the action based on the action
            if package == "nexuslauncher" and action_text == "swipe up" and previous_action == "":
                # First step: Swipe up from the home screen
                if not self.given_rewards["n1"]:
                    reward = intermediate_normal_reward
                    self.given_rewards["n1"] = True
                else:
                    reward = 1
                # Avoid getting rewards for pointless swiping and only rewarding swiping in the home screen
                print("Yay4 first step made!")
            elif action_text == "swipe from top":
                if package == "systemui":
                    if not self.given_rewards["sc2"]:
                        reward = intermediate_short_cut_reward
                        self.given_rewards["sc2"] = True
                    else:
                        reward = 1
                    print("Yay3 second step made!")
                else:
                    if not self.given_rewards["sc1"]:
                        reward = intermediate_short_cut_reward
                        self.given_rewards["sc1"] = True
                    else:
                        reward = 1
                    print("Yay3 first step made!")
            elif action_text == "Settings":
                # Second step: Tap "Settings"
                if not self.given_rewards["n2"]:
                    reward = intermediate_normal_reward
                    self.given_rewards["n2"] = True
                else:
                    reward = 1
                print("Yay4 second step made!")
            elif package == "settings" and action_text == "Network & internet":
                # Third step: Tap "Network & internet"
                if not self.given_rewards["n3"]:
                    reward = intermediate_normal_reward
                    self.given_rewards["n3"] = True
                else:
                    reward = 1
                print("Yay4 third step made!")

        reward -= 1

        # if full exploration mode goes straight to this
        if action_text == "Airplane mode" or action_text == "Airplane mode, Off":
            # Final step: Tap "Airplane mode"
            reward = finish_reward
            print("Yay FINISHED!!")
            done = True

        return reward, done
