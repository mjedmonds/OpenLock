from random import randint

from openlock.scenarios.multi_lock import MultiLockScenario
from openlock.scenarios.CE3 import CommonEffect3Scenario
from openlock.scenarios.CC3 import CommonCause3Scenario
from openlock.scenarios.CE4 import CommonEffect4Scenario
from openlock.scenarios.CC4 import CommonCause4Scenario
from openlock.scenarios.two_step_testing_scenario import TwoStepTestingScenario


TESTING_SCENARIOS = [("CE3", "CE4"), ("CE3", "CC4"), ("CC3", "CE4"), ("CC3", "CC4")]


def select_scenario(scenario_name, use_physics=True):
    scenario_selected = None
    if scenario_name == "CE3" or scenario_name == "CE3_simplified":
        scenario_selected = CommonEffect3Scenario(use_physics=use_physics)
    elif scenario_name == "CC3" or scenario_name == "CC3_simplified":
        scenario_selected = CommonCause3Scenario(use_physics=use_physics)
    elif scenario_name == "CE4":
        scenario_selected = CommonEffect4Scenario(use_physics=use_physics)
    elif scenario_name == "CC4":
        scenario_selected = CommonCause4Scenario(use_physics=use_physics)
    elif scenario_name == "multi-lock":
        scenario_selected = MultiLockScenario()
    elif scenario_name == "TwoStepTestingScenario":
        scenario_selected = TwoStepTestingScenario(use_physics=use_physics)
    else:
        raise ValueError(
            "Invalid scenario chosen in settings_scenario.py: %s" % scenario_name
        )
    return scenario_selected


def select_random_scenarios():
    idx = randint(0, 3)
    return TESTING_SCENARIOS[idx]
