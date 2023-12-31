#Import Game
import gym_super_mario_bros
#Import Joypad (basically gives access to a virtual controller)
from nes_py.wrappers import JoypadSpace
#Use SIMPLE Controls (setting type of actions available)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#Import wrapper for grayscaling
from gym.wrappers import GrayScaleObservation
#Import wrappers for vectorization
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#Import matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
#os for file maintnance
import os
#PPO algo
from stable_baselines3 import PPO
#Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback



#sourced class from youtube
#used for saving as the model trains
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_YVal5_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True



#separated code redone for ease of use

TIME_STEPS = 50000000 #trainging timestep amount
SAVE_STEPS = 1000000 #how often will a save occur? If SAVE_STEPS = TIME_STEPS only save when finished


#create base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#apply grayscale
env = GrayScaleObservation(env, keep_dim=True)
#wrap in dummy environment
env = DummyVecEnv([lambda: env])
#stack frames
env = VecFrameStack(env, 4, channels_order='last')


#pathing for data logs
CHECKPOINT_DIR = '' #logged models and different training points
LOG_DIR = '' #tensor board for info on the model
#setup model saving callback
#check_freq = the amount of iterations between saves (watch storage)
callback = TrainAndLoggingCallback(check_freq=SAVE_STEPS, save_path=CHECKPOINT_DIR)
#creates the neural network and the AI itself (CnnPolicy is best for image processing)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)


#train the model so it can learn
#total_timesteps = how many iterations you want to trian for (should be put in the millions)
model.learn(total_timesteps=TIME_STEPS, callback=callback)