import os

class AirplaneTask:
    """
    Represents a task for enabling airplane mode on an Android emulator.

    This class provides methods for resetting the task and evaluating the rewards based on the interactions with the UI.

    Attributes:
        emulator_id (str): The ID of the emulator where the task is performed.
        current_menu (str): Tracks the current state/menu in the task flow.
    """
    def __init__(self, emulator_id):
        """
        Initializes the AirplaneTask instance.

        Args:
            emulator_id (str): The ID of the emulator.
        """
        self.emulator_id = emulator_id
        self.current_menu = "Home"   # Initial task state

    def reset_task(self):
        """
        Resets the task to its initial state.

        Clears the system settings app and ensures airplane mode is disabled and Wi-Fi is enabled.
        """
        self.current_menu = "Home"
        os.system("adb -s {0} shell pm clear com.android.settings".format(self.emulator_id))
        os.system("adb -s {0} shell settings put global airplane_mode_on 0".format(self.emulator_id))
        os.system("adb -s {0} shell svc wifi enable".format(self.emulator_id))

    def get_reward(self, action_text):
        """
        Evaluates the reward based on the action text.

        Args:
            action (str): The text label of the UI action selected by the agent.

        Returns:
            Tuple[int, bool]: reward, done
        """
        reward = -1
        done = False

        # Evaluate the action based on the action
        if self.current_menu == "Home" and action_text == "swipe up":
            # First step: Swipe up from the home screen
            reward = 2
            # Avoid getting rewards for pointless swiping and only rewarding swiping in the home screen
            self.current_menu = "Main"
            print("Yay first step made!")
        elif action_text == "Settings":
            # Second step: Tap "Settings"
            reward = 5
            print("Yay second step made!")
        elif action_text == "Network & internet":
            # Third step: Tap "Network & internet"
            reward = 10
            print("Yay third step made!")
        elif action_text == "Airplane mode":
            # Final step: Tap "Airplane mode"
            reward = 50
            print("Yay FINISHED!!")
            done = True

        return reward, done