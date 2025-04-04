# AndroidAgent: Navigating Android with Reinforcement Learning

Modern mobile operating systems feature hierarchical user interfaces (UIs) with nested menus that can challenge unexperienced users, compound usability issues and limit accessibility. The vast action space inherent in mobile interfaces including diverse gesture possibilities and the potential for each action to lead to incorrect or suboptimal paths further complicate usability testing. This work addresses these challenges by introducing AndroidAgent, a Reinforcement Learning (RL) framework that leverages Android’s internal view hierarchies. AndroidAgent achieves training success rates above 90\% per episode while converging within a significantly low number of timesteps. The results demonstrate that its constrained action and observation space maintain robustness while reducing irrelevant exploration, even in unguided scenarios. AndroidAgent provides an efficient tool to gain insights for usability, with quantifying the intuitiveness of the interface through metrics such as minimal path length until task completion. This work bridges RL with practical UI, offering developers actionable insights to streamline UI design and enhance user experience in Android environments.

## Getting Started

### Prepare Python environment

Install the python environment as follows:

```shell
$ git clone https://github.com/mj-support/AndroidAgent.git
$ cd AndroidAgent
$ conda env create -f environment.yml
$ conda activate AndroidAgent
```

### Install Android packages

Download and install [Android Studio](https://developer.android.com/studio) as it contains the necessary SDK, ADB and emulator tools. Add ADB and the Android emulator to ```$PATH``` to simplify the usage:

```shell
# for Linux + Bash
$ echo 'export PATH=$PATH:~/Android/Sdk/platform-tools' >> ~/.bashrc
$ echo 'export PATH=$PATH:~/Android/Sdk/emulator' >> ~/.bashrc
$ source ~/.bashrc

# for Mac + Zsh
$ echo 'export PATH=$PATH:~/Library/Android/Sdk/platform-tools' >> ~/.zshrc
$ echo 'export PATH=$PATH:~/Library/Android/Sdk/emulator' >> ~/.zshrc
$ source ~/.zshrc
```

#### Setup Emulator

First you have configure your own Android Virtual Device (AVD). You can see the configured AVDs with ```$ emulator -list-avds```.

1. To configure your own Android Virtual Device (AVD) you need to open Android Studio.
2. Click on 'More Actions' in the 'Welcome to Android Studio'-welcome menu and select 'Virtual Device Manager'.
3. Create a new virtual device by clicking on the '+'
4. Select 'Pixel 2' as hardware, 'VanillaIceCream' as system image and use 'my_avd' as AVD name and click 'Finish'

#### Launch emulator

```shell
$ emulator -avd my_avd -no-boot-anim -netdelay none -no-snapshot -wipe-data -verbose -no-audio -gpu swiftshader_indirect -no-snapshot -read-only -partition-size 512 &

# for headless mode
$ emulator -avd my_avd -no-window -no-boot-anim -netdelay none -no-snapshot -wipe-data -verbose -no-audio -gpu swiftshader_indirect -no-snapshot -read-only -partition-size 512 &

# check your emulator-ID or look different for usable emulators 
$ adb devices
```

## Train the agent
You can start the training with the following command
```shell
$ python3 main.py
```
The resulting model will be saved in the ```models/``` directory. 
So far, only the airplane and youtube task have been implemented, but the code is designed to allow smooth expansion for a wide range of tasks.
