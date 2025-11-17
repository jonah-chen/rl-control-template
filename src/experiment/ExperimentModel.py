from __future__ import annotations

import sys
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']

        self.episode_cutoff = d.get('episode_cutoff', -1)
        self.total_steps = d.get('total_steps')

    @classmethod
    def from_config(cls, cfg: DictConfig | Mapping[str, Any], source: str | None = None):
        """Instantiate from a Hydra/OmegaConf config."""
        if isinstance(cfg, DictConfig):
            # Convert twice so we can drop Hydra-specific settings before resolution.
            raw = OmegaConf.to_container(cfg, resolve=False)
            assert isinstance(raw, dict)
            raw.pop('hydra', None)
            data = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
        else:
            data = deepcopy(cfg)

        assert isinstance(data, dict)
        path = source or data.get('config_path') or data.get('name')
        return cls(data, path)

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
