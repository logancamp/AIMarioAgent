# AIMarioAgent
In this project, my team studied numerous reinforcement learning algorithms and using the OpenAI library created a Mario Agent that was able to complete the first level of Super Mario Bros 64 bit. The algorithms tested included: Proximity Policy Optimization (PPO), Deep Q Learning (DQN), and the Advantage Actor Critic (A2C) algorithms. Due to training times and hardware limitations, however, only the PPO algorithm resulted in a successful completion of the level. <br />

<h2>Languages and Utilities Used</h2>
- <b>Python</b> <br>
- <b>Open AI Libraries</b> <br>
- <b>Tensor Flow</b> <br>
- <b>nes_py</b></b> <br>

<h2>Program walk-through:</h2><br>
Training is done before running the program, our training times went up to 2 days to reach up to 7 million steps. There are two files attached, one is a quick training program which minimized steps in order to train, howver there is also a full walk through program which outlines the process step by step from imports to running the mario agent. There is also a run mario program which can be used to run any of the models trained with these processes. Follow the Full Mario code for more information.
<br>
<br>
There was some minor changes in the positional rewards within the game code aswell: <br><br>
<img src="https://i.imgur.com/Rb3ou1d.jpg" height="80%" width="80%" alt="Updated Library Code"/> 
<br>
This is not necessary but we found that this update to the value function helped smooth out some inconsistencies in our results. To make this change you will have to manually access the imported code after importing the necessary game libraries at the start of the training functions. This should be located in the smb_env.py file within the gym_super_mario_bros library.
<br>
<h2>Output:</h2><br/>
There were a few secitons the agent noticably struggled to get past. The agent was able to complete the full first level only once or twice on the 7 milliion step training model, and only one outlier for the 1 million model. The models tend to be somewhat inconsistent with numerous fails throughout the process, however after some time the model goes further than it originally apears. The model does not complete the course in one run.

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
