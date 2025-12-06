import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import time
import socket
import logging
import csv
import numpy as np
import jax
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from rlglue import RlGlue
from experiment.ExperimentModel import ExperimentModel, load as load_experiment
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from problems.registry import getProblem
from PyExpUtils.results.tools import getParamsAsDict

from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from ml_instrumentation.metadata import attach_metadata
from scheduler import CpuBinder
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# -- Library Configuration --
# ---------------------------
def _require_seed(cfg: DictConfig) -> int:
    seed_value = cfg.get('seed')
    if seed_value is None:
        raise ValueError('Set `seed` in your config; Hydra overrides are optional but the value must exist.')
    return int(seed_value)

@hydra.main(config_path='../conf', config_name='episodic', version_base=None)
def main(cfg: DictConfig):
  with CpuBinder(cpus_per_job=2):
    original_cwd = Path(get_original_cwd())
    os.chdir(original_cwd)

    def _resolve_base_path(path_value: str | None) -> Path:
        if path_value is None:
            return Path(HydraConfig.get().runtime.output_dir)

        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return (original_cwd / candidate).resolve()

    save_root = _resolve_base_path(cfg.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    device = 'gpu' if cfg.gpu else 'cpu'
    jax.config.update('jax_platform_name', device)

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger('exp')
    prod = 'cdr' in socket.gethostname() or cfg.silent
    if not prod:
        logger.setLevel(logging.DEBUG)

    writer = SummaryWriter(log_dir=str(save_root))
    csv_path = save_root / 'metrics.csv'
    csv_file = csv_path.open('w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'return'])

    # ----------------------
    # -- Experiment Def'n --
    # ----------------------
    timeout_handler = TimeoutHandler()

    exp_path = cfg.get('exp_path')
    if exp_path:
        exp = load_experiment(exp_path)
    else:
        exp = ExperimentModel.from_config(cfg.experiment, cfg.experiment.get('config_path'))

    seed = _require_seed(cfg)
    idx = seed

    Problem = getProblem(exp.problem)
    chk = Checkpoint(exp, idx, base_path=cfg.checkpoint_path)
    chk.load_if_exists()
    timeout_handler.before_cancel(chk.save)

    collector = chk.build('collector', lambda: Collector(
        config={
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
            'delta': Pipe(
                MovingAverage(0.99),
                Subsample(100),
            ),
        },
        default=Ignore(),
    ))
    collector.set_experiment_id(idx)

    np.random.seed(seed)

    problem = chk.build('p', lambda: Problem(exp, idx, collector, seed))
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

    glue = chk.build('glue', lambda: RlGlue(agent, env))
    chk.initial_value('episode', 0)

    start_time = time.time()

    if glue.total_steps == 0:
        glue.start()

    for step in range(glue.total_steps, exp.total_steps):
        collector.next_frame()
        chk.maybe_save()
        interaction = glue.step()

        should_end = interaction.term or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff)
        if should_end or exp.total_steps - glue.total_steps <= 1:
            agent.cleanup()

            collector.collect('return', glue.total_reward)
            collector.collect('episode', chk['episode'])
            collector.collect('steps', glue.num_steps)

            chk['episode'] += 1

            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk['episode']
            logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')
            writer.add_scalar('Return', glue.total_reward, global_step=step)
            writer.add_scalar('Steps per Second', fps, global_step=step)
            csv_writer.writerow([step, glue.total_reward])

            if should_end:
                glue.start()
            else:
                break

    collector.reset()
    csv_file.close()

    # Evaluation
    eval_episodes = cfg.get('eval_episodes', 0)
    if eval_episodes > 0:
        logger.info(f'Starting evaluation for {eval_episodes} episodes')
        if hasattr(agent, 'epsilon'):
            setattr(agent, 'epsilon', 0)
        if hasattr(agent, 'update_freq'):
            setattr(agent, 'update_freq', float('inf'))

        eval_returns = []
        for _ in range(eval_episodes):
            glue.start()
            term = False
            while not term:
                interaction = glue.step()
                term = interaction.term
                if exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff:
                    term = True

            eval_returns.append(glue.total_reward)

        writer.add_histogram('Evaluation/Returns', np.array(eval_returns), global_step=exp.total_steps)

        eval_file = save_root / 'eval_returns.txt'
        with eval_file.open('w') as f:
            for ret in eval_returns:
                f.write(f'{ret}\n')

    context = exp.buildSaveContext(idx, base=str(save_root))
    save_path = context.resolve('results.db')
    meta = getParamsAsDict(exp, idx)
    meta |= {'seed': seed}
    attach_metadata(save_path, idx, meta)
    collector.merge(context.resolve('results.db'))
    collector.close()
    chk.delete()


if __name__ == '__main__':
    main()
