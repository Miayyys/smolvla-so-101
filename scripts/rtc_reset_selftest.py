"""Check so101vla/rollout.py's RTC reset fix against LeRobot's real RTC inference thread (policy stubbed, no GPU/arm).

  uv run --frozen python scripts/rtc_reset_selftest.py

The stub policy returns a chunk filled with the observed state value, so every action shows which observation it was
planned from. Two cases, each run with LeRobot's original reset and with the patched one:
  A  hand-back race: reset + resume, then the loop publishes the fresh observation 0.3 s later (a camera read);
  B  in-flight inference: reset while a 0.3 s inference on the old observation is running.
The first action after the reset must come from the fresh observation (2.0), not the old one (1.0).
"""

import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import torch


class Policy:
    config = SimpleNamespace()

    def __init__(self, latency):
        self.latency = latency

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        time.sleep(self.latency)
        return torch.full((1, 50, 6), float(batch['observation.state'].flatten()[0]))

    def reset(self):
        pass


class Processor:
    steps = []

    def __call__(self, x):
        return x

    def reset(self):
        pass


def engine(latency):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    names = [f'j{i}.pos' for i in range(6)]
    e = RTCInferenceEngine(Policy(latency), Processor(), Processor(), SimpleNamespace(robot_type='so_follower'),
                           RTCConfig(execution_horizon=20), {'observation.state': {'dtype': 'float32', 'shape': [6], 'names': names}},
                           'task', 30, 'cpu', rtc_queue_threshold=30, shutdown_event=Event())
    e.start()
    return e, names


def first_action_after_reset(case):
    e, names = engine(0.3 if case == 'B' else 0.02)
    obs = lambda v: {n: v for n in names}
    try:
        e.notify_observation(obs(1.0))
        e.resume()
        while e.get_action(None) is None:  # the old chunk arrives and is consumed
            time.sleep(0.01)
        for _ in range(25):
            e.get_action(None)
        if case == 'A':
            e.pause()  # take over; nothing is published while the human corrects
            time.sleep(0.2)
            e.reset()
            e.resume()
            time.sleep(0.3)  # the loop reads the cameras before it publishes the fresh observation
        else:
            time.sleep(0.05)  # the queue is below the threshold: an inference on the old observation is running
            e.reset()
        e.notify_observation(obs(2.0))
        deadline = time.time() + 5
        while (action := e.get_action(None)) is None and time.time() < deadline:
            time.sleep(0.01)
        return float(action[0])
    finally:
        e.stop()


if __name__ == '__main__':
    from so101vla import rollout
    results = {}
    for patched in (False, True):
        if patched:
            rollout.install_rtc_reset_fix()
        for case in ('A', 'B'):
            results[(patched, case)] = first_action_after_reset(case)
            print(f"{'patched ' if patched else 'original'} reset, case {case}: first action after reset planned from "
                  f"observation {results[(patched, case)]:.1f} (fresh = 2.0)")
    assert results[(True, 'A')] == 2.0 and results[(True, 'B')] == 2.0, results
    print('rtc reset selftest: OK' + ('' if 1.0 in (results[(False, 'A')], results[(False, 'B')])
                                      else ' (the original reset did not show the stale chunk in this run)'))
