
from agents.agent import Agent


class HumanAgent(Agent):

    def __init__(self, params):
        super(HumanAgent, self).__init__(params['data_dir'])

        self.params = params

        participant_id, age, gender, handedness, eyewear, major = self.prompt_subject()

        super(HumanAgent, self).setup_subject(human=True,
                                              participant_id=participant_id,
                                              age=age,
                                              gender=gender,
                                              handedness=handedness,
                                              eyewear=eyewear,
                                              major=major)

    def finish_subject(self, strategy='human', transfer_strategy='human'):
        strategy = self.prompt_strategy()
        transfer_strategy = self.prompt_transfer_strategy()
        super(HumanAgent, self).finish_subject(strategy, transfer_strategy)

    @staticmethod
    def prompt_subject():
        print 'Welcome to OpenLock!'
        participant_id = HumanAgent.prompt_participant_id()
        age = HumanAgent.prompt_age()
        gender = HumanAgent.prompt_gender()
        handedness = HumanAgent.prompt_handedness()
        eyewear = HumanAgent.prompt_eyewear()
        major = HumanAgent.prompt_major()
        return participant_id, age, gender, handedness, eyewear, major

    @staticmethod
    def prompt_participant_id():
        while True:
            try:
                participant_id = int(raw_input('Please enter the participant ID (ask the RA for this): '))
            except ValueError:
                print 'Please enter an integer for the participant ID'
                continue
            else:
                return participant_id

    @staticmethod
    def prompt_age():
        while True:
            try:
                age = int(raw_input('Please enter your age: '))
            except ValueError:
                print 'Please enter your age as an integer'
                continue
            else:
                return age

    @staticmethod
    def prompt_gender():
        while True:
            gender = raw_input('Please enter your gender (\'M\' for male, \'F\' for female, or \'O\' for other): ')
            if gender == 'M' or gender == 'F' or gender == 'O':
                return gender
            else:
                continue

    @staticmethod
    def prompt_handedness():
        while True:
            handedness = raw_input('Please enter your handedness (\'right\' for right-handed or \'left\' for left-handed): ')
            if handedness == 'right' or handedness == 'left':
                return handedness
            else:
                continue

    @staticmethod
    def prompt_eyewear():
        while True:
            eyewear = raw_input('Please enter \'yes\' if you wear glasses or contacts or \'no\' if you do not wear glasses or contacts: ')
            if eyewear == 'yes' or eyewear == 'no':
                return eyewear
            else:
                continue

    @staticmethod
    def prompt_major():
        major = raw_input('Please enter your major: ')
        return major

    @staticmethod
    def prompt_strategy():
        strategy = raw_input('Did you develop any particular technique or strategy to solve the problem? If so, what was your technique/strategy? ')
        return strategy

    @staticmethod
    def prompt_transfer_strategy():
        transfer_strategy = raw_input('If you used a particular technique/strategy, did you find that it also worked when the number of colored levers increased from 3 to 4? ')
        return transfer_strategy

