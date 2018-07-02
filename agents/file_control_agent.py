
from agents.agent import Agent


class FileControlAgent(Agent):

    def __init__(self, params):
        super(FileControlAgent, self).__init__(params['data_dir'])

        self.params = params

        super(FileControlAgent, self).setup_subject(human=False)

    def finish_subject(self, strategy='file-control', transfer_strategy='file-control', agent=None):
        if agent is None:
            agent = self
        super(FileControlAgent, self).finish_subject(strategy, transfer_strategy, agent)

