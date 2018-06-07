
import gym
import sys
import os
import zmq
import jsonpickle
import zlib
import pickle

from agents.file_control_agent import FileControlAgent

from gym_lock.settings_scenario import select_scenario
from session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS

# def exit_handler(signum, frame):
#    print 'saving results.csv'
#    np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
#    exit()


def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


def execute_action(manager, action):
    action = manager.env.action_map[action]
    outcome, reward, done, opt = manager.env.step(action)

    return outcome, reward, done, opt


def finish_action(manager):
    env_reset = manager.finish_action()
    if env_reset:
        print(manager.agent.logger.cur_trial.attempt_seq[-1].action_seq)
    return env_reset


if __name__ == '__main__':

    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS['CE3-CE4']
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS['CC4']
    else:
        setting = sys.argv[1]
        # pass a string or an index
        try:
            params = PARAMS[IDX_TO_PARAMS[int(setting)-1]]
        except Exception:
            params = PARAMS[setting]

    params['data_dir'] = '/tmp/OpenLockLearningResults/subjects'
    params['train_attempt_limit'] = 10000
    params['test_attempt_limit'] = 10000
    params['full_attempt_limit'] = True # run to the full attempt limit, regardless of whether or not all solutions were found
    os.makedirs(params['data_dir'], exist_ok=True)

    # this section randomly selects a testing and training scenario
    # train_scenario_name, test_scenario_name = select_random_scenarios()
    # params['train_scenario_name'] = train_scenario_name
    # params['test_scenario_name'] = test_scenario_name

    agent = FileControlAgent(params)
    scenario = select_scenario(params['train_scenario_name'])
    env = gym.make('arm_lock-v0')
    env.full_attempt_limit = params['full_attempt_limit']
    # create session/trial/experiment manager
    manager = SessionManager(env, agent, params)
    manager.update_scenario(scenario)
    manager.set_action_limit(params['train_action_limit'])
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])

    # setup socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:5555')
    print('Simulator server running...')

    # run in an infinite loop, until file says to finish
    while True:
        receive_msg = recv_zipped_pickle(socket)
        if receive_msg == 'quit':
            send_zipped_pickle(socket, 'quitting')
            break
        elif receive_msg == 'reset':
            agent = FileControlAgent(params)
            manager.agent = agent
            trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                            action_limit=params['train_action_limit'],
                                                            attempt_limit=params['train_attempt_limit'])
            print('Reset simulator with new agent')
            send_zipped_pickle(socket, 'env_reset')
        elif receive_msg == 'get_current_trial':
            send_zipped_pickle(socket, manager.agent.logger.cur_trial)
            print('Sent current trial to client')
        elif receive_msg == 'get_current_scenario_name':
            send_zipped_pickle(socket, manager.env.scenario.name)
            print('Sent current scenario name to client')
        else:
            # action sequence
            if isinstance(receive_msg, dict) and 'action_sequence' in receive_msg.keys():
                env_reset = False
                for action in receive_msg['action_sequence']:
                    print('Executing action {}'.format(action))
                    execute_action(manager, action)
                    env_reset = finish_action(manager)
                send_zipped_pickle(socket, {'env_reset': env_reset, 'attempt': manager.agent.logger.cur_trial.cur_attempt})

                print('Sent attempt sequence to client')
            # single action
            if isinstance(receive_msg, dict) and 'action' in receive_msg.keys():
                action = receive_msg['action']
                print('Executing action {}'.format(action))
                execute_action(manager, action)
                env_reset = finish_action(manager)
                send_zipped_pickle(socket, {'env_reset': env_reset, 'attempt': manager.agent.logger.cur_trial.cur_attempt})

                print('Sent action result to client')
            if isinstance(receive_msg, dict) and 'set_trial' in receive_msg.keys():
                trial_selected = receive_msg['set_trial']
                trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                                action_limit=params['train_action_limit'],
                                                                attempt_limit=params['train_attempt_limit'],
                                                                specified_trial=trial_selected)
                print('Set trial to {}'.format(trial_selected))
                send_zipped_pickle(socket, 'trial selected')

    manager.env.render(manager.env, close=True)          # close the window
    manager.agent.finish_subject()

