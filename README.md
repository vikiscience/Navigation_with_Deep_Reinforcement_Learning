[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: output/score_ref.png
[report]: Report.md

# Navigation with Deep Reinforcement Learning

### Introduction

In this project, an agent is trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic (episode length is 300 steps), and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

Current best result is 16.61:

![image2]

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in a preferred folder, and unzip (or decompress) the file.

### Instructions

Agent training is done using Python 3.6 and PyTorch 0.4.0.

1. Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the working environment.

2. In case that PyTorch v.0.4.0 fails to install via pip (eg. for Windows 10), do it manually:

   `conda install pytorch=0.4.0 -c pytorch`

3. Change the variable `file_name_env` in `const.py` to point at the downloaded Environment accordingly, eg. use the path to `Banana.exe` for Windows 10 (for other OS, see [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb))

4. To test if the installation worked, try:

   `python main.py`

    This will print the Environment info. 

5. To train your own agent with hyperparameter values specified in `const.py`, use the following command:

   `python main.py -e train`

6. To see the reference agent provided in this repo in action, execute:

   `python main.py -e test -r`

7. For other command line options, refer to: 

   `python main.py --help`
   
### Model

The problem of agent navigation was solved by utilizing Deep Reinforcment Learning setting, particularly by implementing DQN and Double DQN.

For model details, see [report].
