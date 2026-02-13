import asyncio
import time
from unittest.mock import MagicMock

import pytest

from skyrl_agent.dispatcher.dispatchers import DISPATCHER_REGISTRY, async_semaphore_dispatcher


class MockTrajectory:
    """Mock trajectory that records call order and supports concurrency tracking."""

    def __init__(self, tracker=None):
        self.calls = []
        self.tracker = tracker

    async def init(self):
        self.calls.append(("init", time.monotonic()))

    async def run(self):
        self.calls.append(("run_start", time.monotonic()))
        if self.tracker is not None:
            self.tracker.enter()
        await asyncio.sleep(0.01)
        if self.tracker is not None:
            self.tracker.exit()
        self.calls.append(("run_end", time.monotonic()))

    async def eval(self):
        self.calls.append(("eval", time.monotonic()))


class ConcurrencyTracker:
    def __init__(self):
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    def enter(self):
        self.current += 1
        if self.current > self.peak:
            self.peak = self.current

    def exit(self):
        self.current -= 1


def build_trajectories(num_instances, num_trajectories, tracker=None):
    trajectories = {}
    for i in range(num_instances):
        inst_id = str(i)
        trajectories[inst_id] = {}
        for t in range(num_trajectories):
            trajectories[inst_id][t] = MockTrajectory(tracker=tracker)
    return trajectories


def build_cfg(num_instances, num_trajectories, max_parallel_agents):
    return {
        "num_instances": num_instances,
        "num_trajectories": num_trajectories,
        "max_parallel_agents": max_parallel_agents,
    }


def test_semaphore_registered():
    assert "async_semaphore" in DISPATCHER_REGISTRY


@pytest.mark.asyncio
async def test_all_trajectories_processed():
    num_instances, num_trajectories = 8, 2
    trajs = build_trajectories(num_instances, num_trajectories)
    cfg = build_cfg(num_instances, num_trajectories, max_parallel_agents=4)

    await async_semaphore_dispatcher(cfg, trajs, "init", "run", "eval")

    for inst_id, inst_trajs in trajs.items():
        for tid, traj in inst_trajs.items():
            stage_names = [c[0] for c in traj.calls]
            assert "init" in stage_names, f"init missing for ({inst_id}, {tid})"
            assert "eval" in stage_names, f"eval missing for ({inst_id}, {tid})"


@pytest.mark.asyncio
async def test_stage_order_per_trajectory():
    num_instances, num_trajectories = 4, 2
    trajs = build_trajectories(num_instances, num_trajectories)
    cfg = build_cfg(num_instances, num_trajectories, max_parallel_agents=4)

    await async_semaphore_dispatcher(cfg, trajs, "init", "run", "eval")

    for inst_id, inst_trajs in trajs.items():
        for tid, traj in inst_trajs.items():
            stage_names = [c[0] for c in traj.calls]
            init_idx = stage_names.index("init")
            run_start_idx = stage_names.index("run_start")
            eval_idx = stage_names.index("eval")
            assert init_idx < run_start_idx < eval_idx, (
                f"Wrong order for ({inst_id}, {tid}): {stage_names}"
            )


@pytest.mark.asyncio
async def test_concurrency_bounded():
    num_instances, num_trajectories = 8, 2
    tracker = ConcurrencyTracker()
    trajs = build_trajectories(num_instances, num_trajectories, tracker=tracker)
    max_parallel = 4
    cfg = build_cfg(num_instances, num_trajectories, max_parallel_agents=max_parallel)

    await async_semaphore_dispatcher(cfg, trajs, "init", "run", "eval")

    assert tracker.peak <= max_parallel, (
        f"Peak concurrency {tracker.peak} exceeded limit {max_parallel}"
    )
    assert tracker.peak > 0, "No concurrency observed"


@pytest.mark.asyncio
async def test_concurrency_bounded_by_total():
    num_instances, num_trajectories = 2, 2
    tracker = ConcurrencyTracker()
    trajs = build_trajectories(num_instances, num_trajectories, tracker=tracker)
    cfg = build_cfg(num_instances, num_trajectories, max_parallel_agents=100)

    await async_semaphore_dispatcher(cfg, trajs, "init", "run", "eval")

    total = num_instances * num_trajectories
    assert tracker.peak <= total, (
        f"Peak concurrency {tracker.peak} exceeded total trajectories {total}"
    )


@pytest.mark.asyncio
async def test_handles_none_init_fn():
    num_instances, num_trajectories = 2, 2
    trajs = build_trajectories(num_instances, num_trajectories)
    cfg = build_cfg(num_instances, num_trajectories, max_parallel_agents=4)

    await async_semaphore_dispatcher(cfg, trajs, None, "run", "eval")

    for inst_id, inst_trajs in trajs.items():
        for tid, traj in inst_trajs.items():
            stage_names = [c[0] for c in traj.calls]
            assert "init" not in stage_names, f"init should not be called for ({inst_id}, {tid})"
            assert "run_start" in stage_names, f"run missing for ({inst_id}, {tid})"
            assert "eval" in stage_names, f"eval missing for ({inst_id}, {tid})"
