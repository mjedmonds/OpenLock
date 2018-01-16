from random import randint

from gym_lock.scenarios.multi_lock import MultiLockScenario
from gym_lock.scenarios.CE3 import CommonEffect3Scenario
from gym_lock.scenarios.CC3 import CommonCause3Scenario
from gym_lock.scenarios.CE4 import CommonEffect4Scenario
from gym_lock.scenarios.CC4 import CommonCause4Scenario


TESTING_SCENARIOS = [
  ('CE3', 'CE4'),
  ('CE3', 'CC4'),
  ('CC3', 'CE4'),
  ('CC3', 'CC4')
]


def select_scenario(scenario, use_physics=True):
    scenario_selected = None
    if scenario == 'CE3':
        scenario_selected = CommonEffect3Scenario(use_physics=use_physics)
    elif scenario == 'CC3':
        scenario_selected = CommonCause3Scenario(use_physics=use_physics)
    elif scenario == 'CE4':
        scenario_selected = CommonEffect4Scenario(use_physics=use_physics)
    elif scenario == 'CC4':
        scenario_selected = CommonCause4Scenario(use_physics=use_physics)
    elif scenario == 'multi-lock':
        scenario_selected = MultiLockScenario()
    else:
        raise ValueError('Invalid scenario chosen in settings_scenario.py: %s' % scenario)
    return scenario_selected


def select_random_scenarios():
    idx = randint(0, 3)
    return TESTING_SCENARIOS[idx]
