from collections import defaultdict, deque
from onpolicy.runner.shared.base_runner import Runner

class abrRunner(Runner):
    def __init__(self, config):
        super(abrRunner, self).__init__(config)
        self.env_infos = defaultdict(list)