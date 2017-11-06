import gym
import signal
import numpy as np
from gym_lock.envs import ArmLockEnv
from gym_lock.common import Action
import time


env = gym.make('arm_lock-v0')

def exit_handler(signum, frame):
   print 'saving results.csv'
   np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
   exit()

signal.signal(signal.SIGINT, exit_handler)

action_script = [Action('push_perp', ('l2', 4)),    # try to unlock l2, but it doesn't budge!
                 Action('pull_perp', ('l2', 4)),    # try pulling l2 instead, still won't budge
                 Action('push_perp', ('l0', 4)),    # unlock l0
                 Action('push_perp', ('l1', 4)),    # unlock l1
                 Action('push_perp', ('l2', 4)),    # try to unlock l2, but it still doesn't budge!
                 Action('pull_perp', ('l2', 4)),    # try pulling l2 instead, it works
                 Action('push_perp', ('door', 4)),  # open the door
                 Action('pull_perp', ('l1', 4)),    # lock l1 (door locks too!)
                 Action('push_perp', ('l2', 4)),    # try to move l2 again
                 Action('pull_perp', ('l2', 4)),    # and it now doesn't work because l1 is locked!
                 Action('push_perp', ('l1', 4)),    # so let's re-unlock l1 (re-unlocks door!)
                 Action('pull_perp', ('door', 1)),  # close the door
                 Action('pull_perp', ('door', 1)),
                 Action('pull_perp', ('door', 1)),
                 Action('push_perp', ('l2', 4)),    # then re-lock l2
                 Action('pull_perp', ('l1', 4)),    # re-lock l1
                 Action('pull_perp', ('l0', 4))]    # re-lock l0



print 'Hello! Welcome to the game!'

#time.sleep(1)

print 'This is a bomb:'

print '  --  '
print ' /  \ '
print '|    |'
print '|    |'
print ' \\  /'
print '  --  '

#time.sleep(1)
print '''See that door on your right? It is the vertical vertical on your right, with the
         red circle (the door hinge) and black circle (it's lock). That is your only escape.'''
#time.sleep(1)
print    '''To open it, you must manipulate the three locks (the rectangles above, below, and
         to your left). Their behavior is unknown! You'll know that you unlocked the door
         when the black circle goes away'''
#time.sleep(1)
print 'ready...'
#time.sleep(1)
print 'set...'
#time.sleep(1)
print 'go!'

env.render()

obs = env.reset()
print obs['OBJ_STATES']
print obs['_FSM_STATE']

while(True):
    env.render()

