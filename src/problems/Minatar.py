from functools import partial
from environments.Minatar import Minatar as MiniAtari
from problems.BaseProblem import BaseProblem
from ml_instrumentation.Collector import Collector
from experiment.ExperimentModel import ExperimentModel

class Minatar(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector, seed: int, game: str):
        super().__init__(exp, idx, collector, seed)

        self.env = MiniAtari(game, self.seed)
        self.actions = self.env.env.num_actions()

        self.observations = self.env.env.state_shape()
        self.gamma = 0.99


Breakout = partial(Minatar, game='breakout')
Seaquest = partial(Minatar, game='seaquest')
Asterix = partial(Minatar, game='asterix')
Freeway = partial(Minatar, game='freeway')
SpaceInvaders = partial(Minatar, game='space_invaders')
