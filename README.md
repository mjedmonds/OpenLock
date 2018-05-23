gym-lock is an OpenAI Gym environment.

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

To run:

1. Inside the gym-lock folder: `python human_open_lock.py`

2. Click around! Every click specifies a (x, y, theta) configuration of the end effector.
   After you click, the an arrow will appear specifying the current target. Sit back for a second
   or two while the inverse kinematics and PID controllers move the arm into the proper configuration.
   Also, if you're touching another object, press enter to attach to it so that you can pull. Press
   enter again to detatch. Detatching looks a bit like Spiderman right now. This will hopefully
   change in the future to something more rigid.

Checkout gym-lock/gym_lock/settings.py for some of the more significant settings that you
can adjust.