import os

class YoutubeTask:
    """
    Represents a task for enabling airplane mode on an Android emulator.

    This class provides methods for resetting the task and evaluating the rewards based on the interactions with the UI.

    Attributes:
        emulator_id (str): The ID of the emulator where the task is performed.
    """
    def __init__(self, emulator_id, token="Charlie bit my finger! ORIGINAL"):
        """
        Initializes the AirplaneTask instance.

        Args:
            emulator_id (str): The ID of the emulator.
        """
        self.emulator_id = emulator_id
        self.token = token.replace('\\', '')

    def reset_task(self):
        """
        Resets the task to its initial state.

        Clears the Youtube app and ensures airplane mode is disabled and Wi-Fi is enabled.
        """
        os.system("adb -s {0} shell pm clear com.google.android.youtube".format(self.emulator_id))
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

        # Evaluate the action based on the action
        if package == "nexuslauncher" and action_text == "swipe up":
            # First step: Swipe up from the home screen
            reward = 2
            # Avoid getting rewards for pointless swiping and only rewarding swiping in the home screen
            print("Yay first step made!")
        elif package == "nexuslauncher" and action_text == "YouTube":
            # Second step: Tap "YouTube"
            reward = 5
            print("Yay second step made!")
        elif (package == "permissioncontroller" and
              (action_text == "Allow" or action_text == "Don’t allow")):
            # Third step: Tap "Network & internet"
            reward = 10
            print("Yay third step made!")
        elif package == "youtube" and action_text == "swipe up" and len(ui_options_current) <= 3:
            reward = 20
            print("Yay fourth step made!")
        elif package == "youtube" and action_text == "Accept all" or action_text == "Reject all":
            reward = 30
            print("Yay fifth step made!")
        elif package == "youtube" and action_text == "Search" or action_text == "Search YouTube":
            reward = 40
            print("Yay sixth step made!")
        elif package == "youtube" and action_text == "Text field Search YouTube":
            reward = 50
            print("Yay seventh step made!")
        elif package == "youtube" and self.token in action_text and action_text != self.token:
            reward = 500
            print("Yay FINISHED!!")
            done = True

        return reward, done
