from gymnasium.envs.registration import register

register(id="Android-v0",
         entry_point="environment.android_env:AndroidEnv")
