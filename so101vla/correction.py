"""Human-correction rollouts on the real SO-101 (HG-DAgger style): ``--strategy.type=correction`` for so101vla.rollout.

Each dataset episode is one task attempt, as in scripts/eval.fish. The policy drives; the operator takes over with the
leader arm where it goes wrong and hands back once the current step is done. Every recorded frame carries
``intervention`` (True = the leader drove the follower). Keys (keep the terminal focused on Wayland):

  space  model -> paused: the policy stops, the leader is driven (2 s) to the follower's pose and holds it
         paused -> model;  correcting -> model (hand back)
  tab    paused -> correcting: leader torque off, the follower follows the leader
         correcting -> paused: both arms hold where they are
  n      end the attempt and save it (also ends the reset phase early);  r  discard and redo it;  q  stop

Paused frames are not recorded, so a correction directly continues the frame sequence. ``--dataset.episode_time_s``
limits recorded seconds per attempt (0 = no limit) and never ends an attempt during a correction. Outside a correction the leader
holds its pose under torque (it never drops when handing back); at the end of an attempt it is driven to the
follower's startup pose and released, and the follower returns (1 s) to that pose while the scene is reset.
Differences from LeRobot's ``dagger`` strategy: per-attempt episodes instead of size-based rotation, autonomous and
correction frames both recorded, and the leader's goal is set to its present position before every torque enable
(Feetech servos keep the last goal while limp and would jump back to it).

``python -m so101vla.correction <dataset_root>`` prints per-episode correction statistics.
"""

import contextlib
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

AUTONOMOUS, PAUSED, CORRECTING = 'autonomous', 'paused', 'correcting'
TRANSITIONS = {(AUTONOMOUS, 'space'): PAUSED, (PAUSED, 'space'): AUTONOMOUS, (CORRECTING, 'space'): AUTONOMOUS,
               (PAUSED, 'tab'): CORRECTING, (CORRECTING, 'tab'): PAUSED}


def install():
    """Register the ``correction`` strategy type and route it through lerobot-rollout's strategy factory."""
    from lerobot.rollout.configs import DAggerStrategyConfig, RolloutStrategyConfig
    from lerobot.rollout.strategies.dagger import DAggerStrategy
    import lerobot.scripts.lerobot_rollout as rollout_script

    # Subclassing the DAgger config keeps LeRobot's checks (teleop required) and its ``intervention`` dataset feature.
    @RolloutStrategyConfig.register_subclass('correction')
    @dataclass
    class CorrectionStrategyConfig(DAggerStrategyConfig):
        pass

    class CorrectionStrategy(DAggerStrategy):
        def setup(self, ctx):
            self._init_engine(ctx)
            self._push_executor = None
            self._lock = threading.Lock()
            self._key = None
            self._flags = {'exit_early': False, 'rerecord': False, 'stop': False}
            self._leader_holding = False
            from lerobot.utils.keyboard_input import create_key_listener
            self._listener = create_key_listener(
                self._on_key, controls_help='space=pause/resume or hand back, tab=start/stop correction, n=next, r=redo, q=quit')

        def _on_key(self, name):
            key = name.lower()
            with self._lock:
                if key in ('space', 'tab'):
                    self._key = key
                elif key in ('n', 'right'):
                    self._flags['exit_early'] = True
                elif key in ('r', 'left'):
                    self._flags.update(rerecord=True, exit_early=True)
                elif key in ('q', 'esc'):
                    self._flags.update(stop=True, exit_early=True)

        def _take_key(self):
            with self._lock:
                key, self._key = self._key, None
            return key

        def _leader_hold_here(self, teleop):
            teleop.send_feedback(teleop.get_action())
            teleop.enable_torque()
            self._leader_holding = True

        def _leader_move_to(self, teleop, target, duration_s=2.0):
            from lerobot.common.control_utils import teleop_smooth_move_to
            self._leader_hold_here(teleop)
            teleop_smooth_move_to(teleop, target, duration_s=duration_s)

        def _leader_release(self, teleop):
            teleop.disable_torque()
            self._leader_holding = False

        def _switch(self, old, new, ctx, follower_pos):
            teleop = ctx.hardware.teleop
            logger.info('correction: %s -> %s', old, new)
            if old == AUTONOMOUS:
                self._engine.pause()
                self._leader_move_to(teleop, follower_pos)
            elif new == CORRECTING:
                self._leader_release(teleop)
            elif old == CORRECTING:
                self._leader_hold_here(teleop)
            if new == AUTONOMOUS:
                self._interpolator.reset()
                self._engine.reset()
                self._engine.resume()

        def run(self, ctx):
            from lerobot.datasets import VideoEncodingManager
            from lerobot.utils.utils import log_say
            cfg = ctx.runtime.cfg
            dataset = ctx.data.dataset
            num_episodes = cfg.dataset.num_episodes
            with VideoEncodingManager(dataset):
                try:
                    saved = 0
                    while saved < num_episodes and not self._flags['stop'] and not ctx.runtime.shutdown_event.is_set():
                        log_say(f'Recording episode {dataset.num_episodes}', cfg.play_sounds)
                        self._attempt(ctx, cfg.dataset.episode_time_s, cfg.dataset.single_task or cfg.task)
                        self._park(ctx)
                        rerecord = self._flags['rerecord']
                        if rerecord:
                            log_say('Re-record episode', cfg.play_sounds)
                            self._flags['rerecord'] = False
                            dataset.clear_episode_buffer()
                        elif dataset.has_pending_frames():
                            dataset.save_episode()
                            saved += 1
                        if not self._flags['stop'] and (saved < num_episodes or rerecord):
                            log_say('Reset the environment', cfg.play_sounds)
                            print(f'correction: {saved}/{num_episodes} attempts saved; rearrange the objects, then press n '
                                  f'to start the next attempt (q stops)', flush=True)
                            self._wait(ctx, cfg.dataset.reset_time_s)
                finally:
                    logger.info('correction: control loop ended - saving any in-progress episode')
                    self._engine.pause()
                    with contextlib.suppress(Exception):
                        if not self._flags['rerecord'] and dataset.has_pending_frames():
                            dataset.save_episode()

        def _attempt(self, ctx, control_time_s, task):
            from lerobot.rollout.strategies.core import send_next_action
            from lerobot.utils.constants import ACTION, OBS_STR
            from lerobot.utils.feature_utils import build_dataset_frame
            from lerobot.utils.robot_utils import precise_sleep
            robot, teleop, dataset = ctx.hardware.robot_wrapper, ctx.hardware.teleop, ctx.data.dataset
            features, processors = ctx.data.dataset_features, ctx.processors
            control_interval = self._interpolator.get_control_interval(ctx.runtime.cfg.fps)
            self._take_key()
            self._flags['exit_early'] = False
            self._interpolator.reset()
            self._engine.reset()
            self._engine.resume()
            phase, hold = AUTONOMOUS, None
            # The limit counts recorded frames only (paused time and the 2 s leader moves are not recorded), 0 = none,
            # and never cuts a correction short: the operator ends attempts with n.
            limit, recorded = control_time_s * ctx.runtime.cfg.fps, 0
            while True:
                loop_start = time.perf_counter()
                if self._flags['exit_early'] or ctx.runtime.shutdown_event.is_set():
                    break
                if 0 < limit <= recorded and phase != CORRECTING:
                    logger.info('correction: %.0f s recorded, attempt ended by the time limit', control_time_s)
                    break
                obs = robot.get_observation()
                key = self._take_key()
                if (phase, key) in TRANSITIONS:
                    new = TRANSITIONS[(phase, key)]
                    follower_pos = {k: v for k, v in obs.items() if k.endswith('.pos')}
                    self._switch(phase, new, ctx, follower_pos)
                    # Paused: the follower holds where it is (measured pose; the last command may not be reached).
                    hold = follower_pos if new == PAUSED and phase == AUTONOMOUS else hold
                    phase = new
                    obs = robot.get_observation()
                    loop_start = time.perf_counter()
                if phase == CORRECTING:
                    obs_processed = processors.robot_observation_processor(obs)
                    action = processors.teleop_action_processor((teleop.get_action(), obs))
                    hold = processors.robot_action_processor((action, obs))
                    robot.send_action(hold)
                    self._log_telemetry(obs_processed, action, ctx.runtime)
                elif phase == PAUSED:
                    robot.send_action(hold)
                    action = None
                else:
                    obs_processed = self._process_observation_and_notify(processors, obs)
                    if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, control_interval):
                        continue
                    action = send_next_action(obs_processed, obs, ctx, self._interpolator)
                    if action is not None:
                        self._log_telemetry(obs_processed, action, ctx.runtime)
                if action is not None:
                    dataset.add_frame({**build_dataset_frame(features, obs_processed, prefix=OBS_STR),
                                       **build_dataset_frame(features, action, prefix=ACTION),
                                       'task': task, 'intervention': np.array([phase == CORRECTING], dtype=bool)})
                    recorded += 1
                dt = time.perf_counter() - loop_start
                if dt > control_interval:
                    logger.warning('correction: loop at %.1f Hz, below %s Hz', 1 / dt, ctx.runtime.cfg.fps)
                precise_sleep(max(control_interval - dt, 0.0))
            self._engine.pause()
            # n/r/q also set exit_early; clear it here or the reset phase that follows would end at once.
            self._flags['exit_early'] = False

        def _park(self, ctx):
            """After an attempt: leader to the startup pose and limp, follower back to the startup pose."""
            teleop, hw = ctx.hardware.teleop, ctx.hardware
            if hw.initial_position:
                self._leader_move_to(teleop, hw.initial_position, duration_s=1.5)
                self._return_to_initial_position(hw, duration_s=1)
            self._leader_release(teleop)

        def _wait(self, ctx, seconds):
            start = time.perf_counter()
            while time.perf_counter() - start < seconds and not ctx.runtime.shutdown_event.is_set():
                if self._flags['exit_early']:
                    break
                time.sleep(0.05)
            self._flags['exit_early'] = False

    original = rollout_script.create_strategy

    def create_strategy(config):
        return CorrectionStrategy(config) if config.type == 'correction' else original(config)

    rollout_script.create_strategy = create_strategy
    return CorrectionStrategy


def runs(flags):
    """[start, end) index pairs of consecutive True values."""
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def summary(root):
    """Per episode: frames, number of corrections, share of human frames, and whether the human finished it."""
    import pandas as pd
    table = pd.concat(pd.read_parquet(p) for p in sorted(Path(root, 'data').rglob('*.parquet'))).sort_values('index')
    if 'intervention' not in table:
        raise ValueError(f'{root} has no intervention column')
    out = {}
    for ep, rows in table.groupby('episode_index'):
        human = np.stack(rows['intervention'].to_numpy()).reshape(-1).astype(bool)
        out[int(ep)] = {'frames': len(human), 'corrections': len(runs(human)),
                        'human_fraction': round(float(human.mean()), 3), 'human_finished': bool(human[-1])}
    return out


if __name__ == '__main__':
    for ep, row in summary(sys.argv[1]).items():
        print(json.dumps({'episode': ep, **row}))
