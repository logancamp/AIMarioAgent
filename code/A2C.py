import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

# ***************************************************************************
# This block will run the mario game with random actions

"""env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(100000): 
    if done: 
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    time.sleep(.01)
env.close()"""
# ***************************************************************************

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
# ***************************************************************************
# This block sets up the environment for ai to learn the game (grayscale and frame stacking)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# ***************************************************************************

state = env.reset()
# state, reward, done, info = env.step([5])

# This comment section shows what the grayscaling and framstacking do, dont need it for learning or then game
"""
plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()
"""

import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):  # Need this class for learning

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# ***************************************************************************
# This block is where the AI learns
"""
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
# This is the AI model started
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, device="cpu") 
# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=7000000, callback=callback)

model.save('thisisatestmodel')
"""
# ***************************************************************************

# ***************************************************************************
# This block playes the game with the model from the training

model = A2C.load('')
state = env.reset()
# Start the game
state = env.reset()
# Loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
# ***************************************************************************