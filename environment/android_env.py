import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.airplane_task import AirplaneTask
import xml.etree.ElementTree as ET
import os
import re
import time


class AndroidEnv(gym.Env):
    """Custom Gymnasium environment to interact with an Android emulator for performing tasks
    such as enabling airplane mode. The agent performs actions based on UI elements.
    """

    def __init__(self, emulator_id="emulator-5554", task="airplane", max_ui_options=16, max_steps_per_episode=5):
        """
        Initializes and setups the Android environment.

        Args:
            emulator_id (str):            The ID of the Android emulator to interact with. Defaults to "emulator-5554".
            task (str):                   The task to perform. Currently, it only supports "airplane" for enabling
                                          airplane mode. Defaults to "airplane".
            max_ui_options (int):         The maximum number of possible UI-options that can be processed.
                                          Defaults to 16.
            max_steps_per_episode (int):  The maximum number of steps performed per episode. Defaults to 5.
        """
        self.emulator_id = emulator_id
        self.max_ui_options = max_ui_options
        self.max_steps_per_episode = max_steps_per_episode
        self.max_text_length = 20

        # By default, the actions are tapping on the UI elements, additional gestures can be added if required.
        self.additional_gestures = {
            "swipe up": False,
            "swipe right": False,   # not implemented
            "swipe down": False,    # not implemented
            "swipe left": False,    # not implemented
            "touch & hold": False,  # not implemented
            "drag & drop": False,   # not implemented
            "type": False           # not implemented
        }

        # Enable specific gestures if the task requires them
        if task == "airplane":
            self.additional_gestures["swipe up"] = True
            self.max_ui_options += 1

        self.ui_options = []    # List of UI elements
        self.obs = {}           # Current observation
        self.current_step = 0   # Current step in the episode

        # Define the action space
        self.action_space = spaces.Discrete(self.max_ui_options)

        # Define the observation space
        self.observation_space = spaces.Dict({
            "index": spaces.Box(low=-1, high=self.max_ui_options - 1, shape=(self.max_ui_options,), dtype=np.int32),
            "text": spaces.Box(low=-1, high=255, shape=(self.max_ui_options, self.max_text_length), dtype=np.int16),
            "history": spaces.Box(low=-1, high=self.max_ui_options - 1, shape=(self.max_steps_per_episode,), dtype=np.int32)
        })

        # Assign the specific task required to access the correct reset and reward method
        if task == 'airplane':
            self.task = AirplaneTask(emulator_id=emulator_id)

        print("Initializing emulator: {0}".format(emulator_id))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.

        Args:
            seed (int, optional): Random seed for the environment. Defaults to None.
            options (dict, optional): Additional reset options. Defaults to None.

        Returns:
            Tuple[dict, dict]: Initial observation and additional info.
        """
        print("Reset")
        self.current_step = 0
        info = {}

        # Force Android-emulator to return to the home screen
        os.system("adb -s {0} shell am start -a android.intent.action.MAIN -c android.intent.category.HOME".format(self.emulator_id))
        self.task.reset_task()  # Reset the specific task
        # Scan the UI-elements and export them into a XML-file
        os.system("adb -s {0} shell uiautomator dump".format(self.emulator_id))
        os.system("adb -s {0} pull /sdcard/window_dump.xml window_emulator.xml".format(self.emulator_id))
        time.sleep(2)

        # Initialize observation values to -1
        self.obs = {
            "index": -1 * np.ones(self.observation_space["index"].shape, dtype=self.observation_space["index"].dtype),
            "text": -1 * np.ones(self.observation_space["text"].shape, dtype=self.observation_space["text"].dtype),
            "history": -1 * np.ones(self.observation_space["history"].shape, dtype=self.observation_space["history"].dtype),
        }

        self._get_obs() # Populate the initial observation
        return self.obs, info

    def step(self, action):
        """
        Perform a step in the environment based on the given action.

        Args:
            action (int): The action chosen by the agent.

        Returns:
            Tuple[dict, float, bool, bool, dict]: Observation, reward, done, truncated and info.
        """
        self.current_step += 1
        print("Step", self.current_step)
        reward = -1

        # Map the action to a UI element
        action, action_text, bounds, action_eval = self._map_action(action)

        if action_eval == "valid":
            reward, done = self.task.get_reward(action_text)    # Get the reward from the task for the chosen action
            if reward > 0:
                self._perform_action(action_text, bounds)   # Perform the action on the emulator if valid
                self._get_obs(action)   # Get new observation
            else:
                action_eval = "wrong"

        if action_text.startswith("swipe"):
            print("{0} action is {1} -> {2} -> Reward: {3}".format(action_eval, action, action_text, reward))
        else:
            print("{0} action is {1} -> TAP {2} -> Reward: {3}".format(action_eval, action, action_text, reward))

        # Mark the episodes as done if negative award is given to force reset
        if reward < 0:
            done = True

        # Terminate the episode if the maximum steps are reached
        if self.current_step == self.max_steps_per_episode:
            done = True

        return self.obs, reward, done, False, {}

    def _get_obs(self, action=None):
        """
        Update the observation space by reading the emulator's current UI state.

        Args:
            action (int, optional): The previous action performed. Defaults to None.
        """
        # Read the XML dump of the current UI
        with open('window_emulator.xml', 'rb') as f:
            xml_data = f.read()
        root = ET.fromstring(xml_data)
        self.ui_options = []    # Clear previous UI options
        self._process_additional_gestures() # Add additional gestures
        self._extract_nodes(root)   # Extract new UI-elements


        index = self.current_step - 1
        if self.current_step != 0:  # Update history if not the first step
            self.obs["history"][index] = action

        # Process and update the UI options
        ui_state = self._process_ui_options()
        self.obs["index"] = ui_state["index"]
        self.obs["text"] = ui_state["text"]

    def _process_additional_gestures(self):
        """Add additional gestures like swiping to the UI options."""
        for gesture in self.additional_gestures:
            if self.additional_gestures[gesture]:
                if gesture == "swipe up":
                    gesture_data = {
                        "index": len(self.ui_options),
                        "text": gesture,
                        "bounds": (540, 960, 540, 200)
                    }
                    self.ui_options.append(gesture_data)

    def _extract_nodes(self, node):
        """
        Recursively extract relevant UI elements from the XML node.

        Data to extract: index, text, resource_id, class, package, content_desc, checkable, checked, clickable,
                         enabled, focusable, scrollable, long-clickable, password, selected, bounds

        Args:
            node (Element): XML node representing a UI element.
        """
        #
        node_text = node.get("text")
        node_resource_id = node.get("resource-id")

        # Add nodes with valid text and resource ID
        # Info: Current resource_id check does not include all possible UI elements that are required for other tasks.
        if node_text != "" and node_text is not None:
            if node_resource_id == "android:id/title" or node_resource_id == "com.google.android.apps.nexuslauncher:id/icon":
                node_index = len(self.ui_options)
                node_bounds = re.findall(r'\d+', node.get("bounds"))
                node_bounds = list(map(int, node_bounds))
                node_data = {
                    "index": node_index,
                    "text": node_text,
                    "bounds": node_bounds
                }
                self.ui_options.append(node_data)

        # # Process child nodes recursively
        for child in node.findall('node'):
            self._extract_nodes(child)

    def _process_ui_options(self):
        """
        Process the UI options and create observation arrays for "index" and "text".

        Returns:
            dict: A dictionary containing the processed "index" and "text" arrays.
        """
        # Initialize arrays with default values (-1)
        index_array = -1 * np.ones((self.max_ui_options,), dtype=np.int32)  # Für "index"
        text_array = -1 * np.ones((self.max_ui_options, self.max_text_length), dtype=np.uint8)  # Für "text"

        # Populate arrays with actual data from UI options
        for i, obj in enumerate(self.ui_options[:self.max_ui_options]):
            index_array[i] = obj["index"]
            encoded_text = self._encode_text(obj["text"])
            text_array[i, :len(encoded_text)] = encoded_text

        ui_options = {
            "index": index_array,
            "text": text_array,
        }
        return ui_options

    def _encode_text(self, text):
        """
        Encode the UI-text into numeric array to allow the processing of the observation_space.

        Args:
            text (str): The text to encode.

        Returns:
            np.ndarray: Encoded text as a numeric array.
        """
        encoded = np.zeros((self.max_text_length,), dtype=np.uint8)
        for i, char in enumerate(text[:self.max_text_length]):
            encoded[i] = ord(char)   # Convert character to ASCII
        return encoded

    def _perform_action(self, action_text, bounds):
        """
        Perform an action (e.g. tap or swipe) on the emulator.

        Args:
            action_text (str): The action to perform (e.g., "swipe up").
            bounds (tuple): The coordinates of the UI-element or for the gesture action.
        """
        if action_text.startswith("swipe"):
            command = "adb -s {4} shell input swipe {0} {1} {2} {3}".format(
                bounds[0], bounds[1], bounds[2], bounds[3], self.emulator_id)
        else:
            coord_x = int((bounds[0] + bounds[2]) / 2)
            coord_y = int((bounds[1] + bounds[3]) / 2)
            command = "adb -s {2} shell input tap {0} {1}".format(coord_x, coord_y, self.emulator_id)

        # Execute the command and update the UI state by extracting the UI-elements on the new screen
        os.system(command)
        time.sleep(0.8)
        os.system("adb -s {0} shell uiautomator dump".format(self.emulator_id))
        time.sleep(0.1)
        os.system("adb -s {0} pull /sdcard/window_dump.xml window_emulator.xml".format(self.emulator_id))
        time.sleep(0.1)

    def _map_action(self, action):
        """
        Maps a discrete action to its corresponding UI-element or gesture.
        Background: Some UI-screens have just a few UI-options compared to the action_space.
        In order to reduce the amount of invalid actions, the actions are mapped to the possible UI-options.

        Args:
            action (int): The discrete action selected by the RL agent,
                          which corresponds to an index in the action space.

        Returns:
            Tuple[int, str, tuple, str]: action, action_text, bounds, action_eval
        """
        # Determine the divisor to scale the action space if it exceeds the number of valid UI options
        map_divisor = self.action_space.n // len(self.ui_options)
        # Adjust the action index when the action space is larger than the UI options
        if map_divisor > 1:
            action = action // map_divisor

        action_text = ""
        bounds = None
        action_eval = "invalid"
        # Iterate over available UI options to find the matching action if available
        for ui_option in self.ui_options:
            if ui_option["index"] == action:
                # Match found: retrieve the text, bounds, and mark the action as valid
                action_text = ui_option["text"]
                bounds = ui_option["bounds"]
                action_eval = "valid"
                break

        # Return the mapped action and associated information
        return action, action_text, bounds, action_eval