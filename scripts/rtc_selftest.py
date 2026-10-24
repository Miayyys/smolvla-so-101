"""Check so101vla/rollout.py's RTC fixes against LeRobot's real RTC inference thread (policy stubbed, no GPU/arm).

  uv run --frozen python scripts/rtc_selftest.py

The stub policy returns chunk step k as (observed state value + 0.01 k), so every action shows which observation it
was planned from and which chunk step it is. Two cases, each run with LeRobot's original code and with the patches:
  A  hand-back race: reset + resume, then the loop publishes the fresh observation 0.3 s later (a camera read);
  B  in-flight inference: reset while a 0.3 s inference on the old observation is running.
The first action after the reset must be step 0 of the chunk planned from the fresh observation (2.00): not from the
old observation (1.xx), and no steps skipped since nothing was executed while it was computed (2.0x).
Also checks ActionQueue directly: with 3 actions consumed during an inference, exactly 3 are skipped.
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
        value = float(batch['observation.state'].flatten()[0])
        return (value + 0.01 * torch.arange(50, dtype=torch.float32)).view(1, 50, 1).repeat(1, 1, 6)

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


def queue_skip(consumed):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    q = ActionQueue(RTCConfig(execution_horizon=20))
    chunk = lambda v: (v + 0.01 * torch.arange(50, dtype=torch.float32)).view(50, 1).repeat(1, 6)
    q.merge(chunk(1.0), chunk(1.0), 0, 0)
    before = q.get_action_index()
    for _ in range(consumed):
        q.get()
    q.merge(chunk(2.0), chunk(2.0), 11, before)  # latency estimate 11 steps
    return round((float(q.get()[0]) - 2.0) / 0.01)


if __name__ == '__main__':
    from so101vla import rollout
    results = {}
    for patched in (False, True):
        if patched:
            rollout.install_rtc_fixes()
        name = 'patched ' if patched else 'original'
        for case in ('A', 'B'):
            results[(patched, case)] = value = first_action_after_reset(case)
            print(f'{name} case {case}: first action after reset = {value:.2f} -> planned from observation {int(value)}, '
                  f'chunk step {round((value - int(value)) / 0.01)} (want observation 2, step 0)')
        for consumed in (0, 3):
            results[(patched, consumed)] = skipped = queue_skip(consumed)
            print(f'{name} queue: {consumed} actions consumed during inference -> {skipped} skipped (want {consumed})')
    assert all(abs(results[(True, c)] - 2.0) < 1e-6 for c in 'AB'), results
    assert results[(True, 0)] == 0 and results[(True, 3)] == 3, results
    print('rtc selftest: OK')
