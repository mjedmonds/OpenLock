gym-lock is an OpenAI Gym environment.

To install:

1. Use Python 2

2. Make sure you meet all the pre-requisites specified in the `setup.py` file.
   Specifically, make sure you install pybox2d from source (https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md)

3. Inside the gym-lock folder: `python setup.py install`

4. You may run into problems with graphviz and future modules. Make sure before you run the previous setup step to `sudo apt-get install graphviz-dev` and `sudo apt-get install graphviz`, and if you run into "ImportError: No module named 'future'", run `pip install future`.

To run:

1. Inside the gym-lock folder: `python test.py`

2. Click around! Every click specifies a (x, y, theta) configuration of the end effector.
   After you click, the an arrow will appear specifying the current target. Sit back for a second
   or two while the inverse kinematics and PID controllers move the arm into the proper configuration.
   Also, if you're touching another object, press enter to attach to it so that you can pull. Press
   enter again to detatch. Detatching looks a bit like Spiderman right now. This will hopefully
   change in the future to something more rigid.

If you look at test.py, you'll see the step function which returns obs, rew, done, info.
See the docs in `ArmLockEnv._step()` which is in gym-lock/gym_lock/envs/arm_lock_env.py. 

To start, I've created a simple door lock environment. The top rectangle is the door, the bottom
rectangle is the lock. Try to open the door first and it won't budge. Instead, push the lock a
bit to the right and try opening the door again. It should now open! You cannot yet relock the door.

Checkout gym-lock/gym_lock/settings.py for some of the more significant settings that you
can adjust.

I'm sure there's bugs, just add them to the github issue tracker and I'll get them taken care of!

-Colin
