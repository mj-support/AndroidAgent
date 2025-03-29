import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.airplane_task import AirplaneTask
from environment.youtube_task import YoutubeTask
import xml.etree.ElementTree as ET
import os
import re
import time


class AndroidEnv(gym.Env):
    """Custom Gymnasium environment to interact with an Android emulator for performing RL tasks
    such as enabling airplane mode. The agent performs actions based on retrieved UI elements.
    """

    def __init__(self, emulator_id="emulator-5554", task="airplane", exploration_mode="full_exploration", episode_timesteps=100, max_current_ui_options=20):
        """
        Initializes and setups the Android environment.

        Args:
            emulator_id (str):              The ID of the Android emulator to interact with. Defaults to "emulator-5554".
            task (str):                     The task to perform. Currently, it only supports the "airplane" or "youtube".
                                            Defaults to "airplane".
            exploration_mode (str):         Exploration mode of the agent, also affecting the reward structure.
                                            Possible modes: "guided_restricted", "guided_open", "full_exploration".
                                            Defaults to "full_exploration"
            episode_timesteps (int):        The maximum number of steps performed per episode. Defaults to 100.
            max_current_ui_options (int):   The maximum number of possible UI-options that can be processed.
                                            Defaults to 20.
        """
        self.emulator_id = emulator_id
        self.max_current_ui_options = max_current_ui_options
        self.episode_timesteps = episode_timesteps
        self.max_text_length = 20
        self.max_total_ui_options = 25000
        self.exploration_mode = exploration_mode

        # By default, the actions are tapping on the UI elements, additional gestures can be added if required.
        self.additional_gestures = {
            "swipe up": False,
            "swipe right": False,       # not implemented
            "swipe down": False,        # not implemented
            "swipe from top": False,
            "swipe left": False,        # not implemented
            "touch & hold": False,      # not implemented
            "drag & drop": False,       # not implemented
            "type": False               # not implemented
        }

        # Enable specific gestures if the task requires them
        # Assign the specific task required to access the correct reset and reward method
        if task == "airplane":
            self.additional_gestures["swipe up"] = True
            self.additional_gestures["swipe from top"] = True
            self.max_current_ui_options += 1
            self.token = "airplane"
            self.task = AirplaneTask(emulator_id=emulator_id, token=self.token, exploration_mode=self.exploration_mode, episode_timesteps=self.episode_timesteps)
        if task == "youtube":
            self.additional_gestures["swipe up"] = True
            self.max_current_ui_options += 1
            self.token = "Charlie\ bit\ my\ finger!\ ORIGINAL"
            self.task = YoutubeTask(emulator_id=emulator_id, token=self.token, exploration_mode=self.exploration_mode, episode_timesteps=self.episode_timesteps)

        self.obs = {}
        self.obs_history = []# Current observation
        self.ui_options_total = self._process_additional_gestures()   # List of UI elements
        self.ui_options_current = []
        self.current_step = 0   # Current step in the episode
        self.episode_rewards = 0

        # Define the action space
        self.action_space = spaces.Discrete(self.max_current_ui_options)

        # Define the observation space
        self.observation_space = spaces.Dict({
            "ui_options": spaces.MultiDiscrete([self.max_total_ui_options] * self.max_current_ui_options),  # Aktuelle UI-Optionen
            "history": spaces.MultiDiscrete([self.max_total_ui_options] * self.episode_timesteps)  # Vergangene Aktionen
        })

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
        print("Reset, Length: ", len(self.ui_options_total))
        self.current_step = 0
        self.episode_rewards = 0
        info = {}
        self.obs_history = [{}]
        self.ui_options_current = self._process_additional_gestures()

        # Force Android-emulator to return to the home screen
        os.system("adb -s {0} shell input keyevent KEYCODE_HOME".format(self.emulator_id))
        time.sleep(0.2)
        self.task.reset_task()  # Reset the specific task
        # Scan the UI-elements and export them into a XML-file
        os.system("adb -s {0} shell uiautomator dump".format(self.emulator_id))
        os.system("adb -s {0} pull /sdcard/window_dump.xml window_emulator_{0}.xml".format(self.emulator_id))
        time.sleep(2)

        self.obs = {
            "ui_options": np.zeros(self.max_current_ui_options, dtype=np.int32),
            "history": np.zeros(self.episode_timesteps, dtype=np.int32)
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
        done = False

        # Map the action to a UI element
        action, action_text, bounds, action_eval = self._map_action(action)
        self.obs_history[-1]["action"] = action
        self.obs_history[-1]["action_text"] = action_text
        self.obs_history[-1]["ui_option_id"] = None

        if action_eval == "valid":
            self.obs_history[-1]["ui_option_id"] = self.ui_options_current[action]["id"]
            reward, done = self.task.get_reward(self.obs_history, self.ui_options_current)

            if action_text != "Power menu" and action_text != "Emergency":
                if reward >= 0 or self.exploration_mode == "full_exploration" or self.exploration_mode == "guided_open":
                    self._perform_action(bounds)  # Perform the action on the emulator if valid
                    self._get_obs()  # Get new observation
                elif self.exploration_mode == "guided_restricted":
                    action_eval = "wrong"

        if action_text.startswith("swipe"):
            print("{0} action is {1} -> {2} -> Reward: {3}".format(action_eval, action, action_text, reward))
        else:
            print("{0} action is {1} -> TAP {2} -> Reward: {3}".format(action_eval, action, action_text, reward))

        self.episode_rewards += reward

        # Terminate the episode if the maximum steps are reached
        if self.current_step == self.episode_timesteps:
            done = True

        return self.obs, reward, done, False, {}

    def _process_additional_gestures(self):
        """
        Add additional gestures like swiping to the UI options.
        
        Returns:
            gesture_list (list): List of additional gestures.
        """
        
        gesture_list = []
        for gesture in self.additional_gestures:
            if self.additional_gestures[gesture]:
                if gesture == "swipe up":
                    gesture_data = {
                        "id": len(gesture_list) + 1,
                        "text": gesture,
                        "bounds": (540, 960, 540, 200)
                    }
                    gesture_list.append(gesture_data)
                elif gesture == "swipe from top":
                    gesture_data = {
                        "id": len(gesture_list) + 1,
                        "text": gesture,
                        "bounds": (540, 0, 540, 960)
                    }
                    gesture_list.append(gesture_data)

        return gesture_list

    def _get_obs(self):
        """
        Update the observation space by reading the emulator's current UI state.
        """
        # Read the XML dump of the current UI
        try:
           with open('window_emulator_{0}.xml'.format(self.emulator_id), 'rb') as f:
                xml_data = f.read()
        except:
            os.system("adb -s {0} shell uiautomator dump".format(self.emulator_id))
            time.sleep(10)
            os.system("adb -s {0} pull /sdcard/window_dump.xml window_emulator_{0}.xml".format(self.emulator_id))
            time.sleep(10)
            with open('window_emulator_{0}.xml'.format(self.emulator_id), 'rb') as f:
                xml_data = f.read()
            print("Exception")

        root = ET.fromstring(xml_data)

        self.ui_options_current = self._process_additional_gestures()    # Clear previous UI options # Add additional gestures
        self._extract_nodes(root)   # Extract new UI-elements

        # Process and update the UI options
        self.obs["ui_options"] = np.zeros(self.max_current_ui_options, dtype=np.int32)
        index = 0
        for ui_option in self.ui_options_current[:self.max_current_ui_options]:
            self.obs["ui_options"][index] = ui_option["id"]
            index += 1

        if self.current_step == 0:
            self.obs_history[-1]["package"] = root.find('node').get("package").split(".")[-1]
        else:  # Update history if not the first step
            self.obs_history[-1]["ui_options"] = self.obs["ui_options"]
            self._get_menu_history()
            self.obs_history.append({"package": root.find('node').get("package").split(".")[-1]})

    def _extract_nodes(self, node):
        """
        Recursively extract relevant UI elements from the XML node.

        Possible data to extract: index, text, resource_id, class, package, content_desc, checkable, checked, clickable,
                         enabled, focusable, scrollable, long-clickable, password, selected, bounds

        Args:
            node (Element): XML node representing a UI element.
        """
        node_clickable = node.get("clickable")

        if node_clickable == "true" and not node.get("resource-id").endswith("clock"):
            element_name = self._extract_element_name(node)
            node_bounds = re.findall(r'\d+', node.get("bounds"))
            node_bounds = tuple(map(int, node_bounds))
            node_package = node.get("package").split(".")[-1]

            node_class = node.get("class")
            if "EditText" in node_class:
                element_name = f"Text field {element_name}"

            if node_bounds != (0, 0, 0, 0):
                node_data = {
                    "id": len(self.ui_options_total) + 1,
                    "text": element_name,
                    "bounds": node_bounds,
                    "package": node_package,
                }

                duplicate = False
                for ui_option in self.ui_options_total:
                    # ui_option = self._process_ui_options()
                    if ui_option["text"] == element_name and ui_option["package"] == node_package:
                        duplicate = True
                        node_data["id"] = ui_option["id"]
                        break

                if not duplicate:
                    self.ui_options_total.append(node_data)
                self.ui_options_current.append(node_data)

        # # Process child nodes recursively
        for child in node.findall('node'):
            self._extract_nodes(child)

    def _extract_element_name(self, node):
        """
        Extracts the name of a UI element from the XML node.

        Args:
            node (Element): XML node representing a UI element.
            
        Returns:
            element_name (str): name of the UI element.
        """
        element_name = ""
        node_text = node.get("text")
        node_content_desc = node.get("content-desc")

        if node_text != "":
            element_name = node_text
        elif node_content_desc != "":
            element_name = node_content_desc
        else:
            for child in node.findall('node'):
                element_name = self._extract_element_name(child)
                if element_name != "":
                    break

        return element_name

    def _perform_action(self, bounds):
        """
        Perform an action (e.g. tap or swipe) on the emulator.

        Args:
            bounds (tuple): The coordinates of the UI-element or for the gesture action.
        """
        action_text = self.obs_history[-1]["action_text"]
        if action_text.startswith("swipe"):
            os.system("adb -s {4} shell input swipe {0} {1} {2} {3}".format(bounds[0], bounds[1], bounds[2], bounds[3], self.emulator_id))
            time.sleep(1)
        elif action_text.startswith("Text field"):
            os.system("adb -s {1} shell input text '{0}'".format(self.token, self.emulator_id))
            os.system("adb -s {0} shell input keyevent ENTER".format(self.emulator_id))
        else:
            coord_x = int((bounds[0] + bounds[2]) / 2)
            coord_y = int((bounds[1] + bounds[3]) / 2)
            os.system("adb -s {2} shell input tap {0} {1}".format(coord_x, coord_y, self.emulator_id))

        # Execute the command and update the UI state by extracting the UI-elements on the new screen
        time.sleep(1)
        os.system("adb -s {0} shell uiautomator dump".format(self.emulator_id))
        time.sleep(0.3)
        os.system("adb -s {0} pull /sdcard/window_dump.xml window_emulator_{0}.xml".format(self.emulator_id))
        time.sleep(0.5)

    def _map_action(self, action):
        """
        Maps a discrete action to its corresponding UI-element or gesture.
        Background: Some UI-screens have just a few UI-options compared to the action_space.
        In order to reduce the amount of invalid actions, the actions are mapped to the possible UI-options.

        Args:
            action (int): The discrete action selected by the RL agent, which corresponds to an index in the action space.

        Returns:
            Tuple[int, str, tuple, str]: action, action_text, bounds, action_eval
        """
        # Determine the divisor to scale the action space if it exceeds the number of valid UI options
        if action > len(self.ui_options_current) and len(self.ui_options_current) < self.max_current_ui_options:
            if self.max_current_ui_options // len(self.ui_options_current) > 1:
                action = action % len(self.ui_options_current)

        if action < len(self.ui_options_current):
            action_text = self.ui_options_current[action]["text"]
            bounds = self.ui_options_current[action]["bounds"]
            action_eval = "valid"
        else:
            action_text = ""
            bounds = None
            action_eval = "invalid"

        return action, action_text, bounds, action_eval

    def _get_menu_history(self):
        """
        Store the shortest path of the menu history to avoid repetition
        """
        current_ui_option_id = self.obs_history[-1]["ui_option_id"]
        index = 0
        repetition = False

        for history_ui_option_id in self.obs["history"]:
            if not repetition:
                if history_ui_option_id == 0:
                    self.obs["history"][index] = current_ui_option_id
                    break
                elif history_ui_option_id == current_ui_option_id:
                    for history in self.obs_history:
                        if (history["ui_option_id"] == current_ui_option_id and
                                np.array_equal(history["ui_options"], self.obs["ui_options"])):
                            repetition = True
                            break
                    if not repetition:
                        self.obs["history"][index] = current_ui_option_id
            index += 1

    def _process_ui_options(self):
        """
        Process the UI options and create observation arrays for "index" and "text".

        Returns:
            ui_options (dict): A dictionary containing the processed "index" and "text" arrays.
        """
        # Initialize arrays with default values (-1)
        text_array = -1 * np.ones((self.max_current_ui_options, self.max_text_length), dtype=np.uint8)  # Für "text"

        # Populate arrays with actual data from UI options
        for i, obj in enumerate(self.ui_options[:self.max_current_ui_options]):
            encoded_text = self._encode_text(obj["text"])
            text_array[i, :len(encoded_text)] = encoded_text

        ui_options = {
            "text": text_array,
        }
        return ui_options

    def _encode_text(self, text):
        """
        Additional feature to cncode the UI-text into numeric array to allow the processing of the observation_space.

        Args:
            text (str): The text to encode.

        Returns:
            encoded (np.ndarray): Encoded text as a numeric array.
        """
        encoded = np.zeros((self.max_text_length,), dtype=np.uint8)
        for i, char in enumerate(text[:self.max_text_length]):
            encoded[i] = ord(char)   # Convert character to ASCII
        return encoded
