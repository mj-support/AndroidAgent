# AndroidAgent: Task completion in Android OS using Reinforcement Learning

## Getting Started

### Installation

Install the python environment as follows:

```shell
$ git clone https://github.com/mj-support/AndroidAgent.git
$ cd AndroidAgent
$ conda env create -f environment.yml
$ conda activate AndroidAgent
```

### Prepare Android environment

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

To reconstruct the airplane-task, please follow the next steps. You can also configure your own Android Virtual Device (AVD) with your own settings instead, but in this case a successful agent can no longer be guaranteed. You can see the configured AVDs with ```$ emulator -list-avds```.

1. To configure your own Android Virtual Device (AVD) you need to open Android Studio.
2. Click on 'More Actions' in the 'Welcome to Android Studio'-welcome menu and select 'Virtual Device Manager'.
3. Create a new virtual device by clicking on the '+'
4. To reconstruct the airplan task: select 'Pixel 2' as hardware, 'VanillaIceCream' as system image and use 'my_avd' as AVD name and click 'Finish'

#### Launch emulator

```shell
$ emulator -avd Medium_Phone_API_35 -no-boot-anim -netdelay none -no-snapshot -wipe-data -verbose -no-audio -gpu swiftshader_indirect -no-snapshot -read-only -partition-size 512 &

# for headless mode
$ emulator -avd Medium_Phone_API_35 -no-window -no-boot-anim -netdelay none -no-snapshot -wipe-data -verbose -no-audio -gpu swiftshader_indirect -no-snapshot -read-only -partition-size 512 &

# check your emulator-ID or look different for usable emulators 
$ adb devices
```

## Train the agent
You can start the training with the following command
```shell
$ python3 main.py
```
The resulting model will be saved in the ```models/``` directory. 
So far, only the airplane task has been implemented, but the code is designed to allow smooth expansion for a wide range of tasks.
