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

### Setup & launch emulator
Per default ```Medium_Phone_API_35``` is already installed as an Android Virtual Device (AVD) but you can still configure your own device with Android Studio via the "Virtual Device Manager".
You can see the installed AVDs with ```$ emulator -list-avds```.
Use the following command to start the emulator based on the configured AVD.

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
