import os

class AirplaneTask:
    """
    Represents a task for enabling airplane mode on an Android emulator.

    This class provides methods for resetting the task and evaluating the rewards based on the interactions with the UI.

    Attributes:
        emulator_id (str): The ID of the emulator where the task is performed.
    """
    def __init__(self, emulator_id, token=None, exploration_mode="full_exploration"):
        """
        Initializes the AirplaneTask instance.

        Args:
            emulator_id (str): The ID of the emulator.
        """
        self.emulator_id = emulator_id
        self.token = token
        self.exploration_mode = exploration_mode

    def reset_task(self):
        """
        Resets the task to its initial state.

        Clears the system settings app and ensures airplane mode is disabled and Wi-Fi is enabled.
        """
        os.system("adb -s {0} shell pm clear com.android.settings".format(self.emulator_id))
        os.system("adb -s {0} shell settings put global airplane_mode_on 0".format(self.emulator_id))
        os.system("adb -s {0} shell svc wifi enable".format(self.emulator_id))

    def get_reward(self, obs_history, ui_options_current):
        """
        Evaluates the reward based on the action text.

        Args:
            action_text (str): The text label of the UI action selected by the agent.
            current_menu (dict): The current menu where the action will be performed at.

        Returns:
            Tuple[int, bool]: reward, done
        """
        reward = -1
        done = False
        package = obs_history[-1]['package']
        action_text = obs_history[-1]['action_text']
        previous_action = ""

        if len(obs_history) > 1:
            previous_action = obs_history[-2]['action_text']

        if self.exploration_mode == "guided_restricted" or self.exploration_mode == "guided_open":
            # Evaluate the action based on the action
            if package == "nexuslauncher" and action_text == "swipe up" and previous_action == "":
            #if current_menu == "Home" and action_text == "swipe up":
                # First step: Swipe up from the home screen
                reward = 2
                # Avoid getting rewards for pointless swiping and only rewarding swiping in the home screen
                print("Yay4 first step made!")
            elif action_text == "swipe from top":
                if package == "systemui":
                    for ui_option in ui_options_current:
                        if ui_option["text"] == "Airplane mode, Off":
                            reward = -1
                            break
                    else:
                        reward = 12
                    print("Yay3 second step made!")
                else:
                    reward = 2
                    print("Yay3 first step made!")
            elif action_text == "Settings":
                # Second step: Tap "Settings"
                reward = 4
                print("Yay4 second step made!")
            elif package == "settings" and action_text == "Network & internet":
                # Third step: Tap "Network & internet"
                reward = 8
                print("Yay4 third step made!")

        # if full exploration mode goes straight to this
        if action_text == "Airplane mode" or action_text == "Airplane mode, Off":
            # Final step: Tap "Airplane mode"
            reward = 100
            print("Yay FINISHED!!")
            done = True

        return reward, done
