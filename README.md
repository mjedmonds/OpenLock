# OpenLock

OpenLock is an OpenAI Gym environment, designed for transfer learning. The environment is governed by compositional structure that requires reasoning about an abstract latent state. OpenLock is a virtual "escape room" where agents are required to interact with levers in order to open the door. Agents are required to find _all_ solutions within a room. After completing a single room, agents are moved to a new room with the same underlying abstract structure, but the positions of each lever has been changed. 

This experimental setup is designed to test whether or not agents are capable of forming an abstract representation of the task.

The gif below summarizes the environment's execution and how lever positions change between rooms:

<center><img src="http://www.mjedmonds.com/projects/OpenLock/CogSci18_openlock_solutions.gif" alt="OpenLock environment executions" width="600"></center>

The environment supports a number of scenarios, each of which encode a specific locking mechanism that governs the environment:

<center><img src="http://www.mjedmonds.com/projects/OpenLock/causal_structures.png" alt="causal structures" width="400"></center>

For additional details on the environment, please see the project page for our CogSci 2018 paper: [http://www.mjedmonds.com/projects/OpenLock/CogSci18_OpenLock_CausalRL.html](http://www.mjedmonds.com/projects/OpenLock/CogSci18_OpenLock_CausalRL.html)

## Installation
To install:

1. Use Python 3.5+

2. Install the following system-level packages:
```
sudo apt-get install python3-tk graphviz graphviz-dev
```

2. Create a virtual envrionment for OpenLock. Then run:
```
pip3 install -r requirements.txt
```

3. Finally, install pybox2d from source (https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md)

4. You may run into problems with modules. If you run into "ImportError: No module named 'future'", run `pip3 install future`.

## Bibtex
If you use this environment in your work, please use the following citation:
```
@inproceedings{edmonds2018human,
  title={Human Causal Transfer: Challenges for Deep Reinforcement Learning},
  author={Edmonds, Mark and Kubricht, James, Feng and Summers, Colin and Zhu, Yixin and Rothrock, Brandon and Zhu, Song-Chun and Lu, Hongjing},
  booktitle={40th Annual Meeting of the Cognitive Science Society},
  year={2018}
}
```