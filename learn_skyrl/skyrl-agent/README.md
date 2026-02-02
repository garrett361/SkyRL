# SkyRL Agent Architecture

## Overview

`skyrl-agent` is a multi-turn agent framework that provides:
- Modular agent implementations (ReAct, CodeAct)
- Pluggable inference backends (OpenAI, VeRL, Tinker)
- Tool execution and message history management
- RL training integration via VeRL

## Directory Structure

```
skyrl_agent/
├── agents/           # Agent implementations and trajectory wrappers
├── integrations/     # Backend abstraction (OpenAI, VeRL, Tinker, skyrl-train)
├── functional/       # Data structures, history, encoding utilities
├── dispatcher/       # Async execution orchestration
├── tools/            # Tool registry and implementations
├── tasks/            # Task-specific logic (instructions, evaluation)
└── config/           # Configuration utilities
```

## Core Agent Loop

The agent loop is organized into three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  AgentRunner (agents/base.py:113)                                   │
│  - Orchestrates batches of trajectories                             │
│  - Calls dispatcher for parallel execution                          │
│  - Methods: run(), _initialize_trajectories(), _post_process()      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BaseTrajectory / ReActTrajectory (agents/base.py:76)               │
│  - Wraps a single agent execution                                   │
│  - Methods: initialize_trajectory(), generate_trajectory(),         │
│             evaluate_trajectory()                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ReActAgent (agents/react/react_agent.py:39)                        │
│  - Implements the actual agent loop                                 │
│  - Methods: run(), step(), _prepare_llm_input(), _execute_tool()    │
│  - Uses @record_transition decorator to capture (obs, action, rew)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Step Loop (ReActAgent.step)

```
┌──────────────────────┐
│ _prepare_llm_input() │  Encode MessageHistory → input_ids
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ @record_transition   │  LLM generation via backend.async_generate_ids()
│ (captures obs/action)│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ parse_tool_call()    │  Extract tool name and arguments from response
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ _execute_tool()      │  TOOL_REGISTRY[name].call(args, agent=self)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ _append_tool_output()│  Add tool response to MessageHistory
└──────────┴───────────┘
           │
           ▼ (repeat until done or max_iterations)
```

## Agent Implementations: ReAct vs OHCodeAct

SkyRL provides two agent implementations with different design philosophies:

### ReActAgent (`agents/react/react_agent.py:39`)

A **standalone, lightweight** implementation built entirely within SkyRL:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ReActAgent                                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Dependencies: SkyRL only (no external agent frameworks)            │
│  Message mgmt: MessageHistory, MessageEncoder                       │
│  Tool system:  TOOL_REGISTRY (finish, sandbox_fusion, web_browser)  │
│  RL support:   @record_transition → Transition objects              │
│  Use cases:    General-purpose tasks, math, web research            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- Self-contained message history management via `MessageHistory`
- Incremental token encoding via `MessageEncoder` (avoids re-tokenizing entire history)
- Records `Transition` objects with `(ob, ac, reward, done)` for RL training
- Uses `@record_transition` decorator to capture LLM inputs/outputs
- Works with any `BaseTask` via generic interface

### OHCodeActAgent (`agents/oh_codeact/codeact_agent.py:47`)

An **OpenHands-integrated** agent that extends `CodeActAgent`:

```
┌─────────────────────────────────────────────────────────────────────┐
│  OHCodeActAgent extends openhands.CodeActAgent                      │
├─────────────────────────────────────────────────────────────────────┤
│  Dependencies: OpenHands framework (runtime, events, actions)       │
│  Message mgmt: OpenHands State + internal self.messages list        │
│  Tool system:  OpenHands function calling + sandboxed runtime       │
│  RL support:   Messages only (no Transition recording)              │
│  Use cases:    SWEBench, code editing, sandboxed execution          │
└─────────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- Extends OpenHands' `CodeActAgent` class (`codeact_agent.py:47`)
- Uses OpenHands event-action model (`State`, `Event`, `Action`, `AgentFinishAction`)
- Manages sandboxed runtime for code execution (`runtime.event_stream`)
- Converts between OpenHands function call format and SkyRL format
- Tightly coupled to `SWEBenchTask` (`codeact_runner.py:109`)

### Comparison

| Aspect | ReActAgent | OHCodeActAgent |
|--------|------------|----------------|
| **Framework** | Standalone (SkyRL only) | Extends OpenHands |
| **Message storage** | `MessageHistory` class | `self.messages` list + OpenHands `State` |
| **Tool execution** | `TOOL_REGISTRY[name].call()` | OpenHands runtime + function calling |
| **RL training** | `Transition` objects with logprobs | Messages only |
| **Runtime** | Lightweight (no sandbox) | Full OpenHands sandbox |
| **Task coupling** | Generic `BaseTask` | Tightly coupled to `SWEBenchTask` |
| **Step method** | `async step()` returns `(done, reason, result)` | `step(state)` returns `Action` |

### Why Both Exist

1. **ReActAgent** is the general-purpose workhorse:
   - Simpler, more portable, easier to debug
   - Native RL training support via `Transition` recording
   - Works with any task type (math, web research, general QA)

2. **OHCodeActAgent** is specialized for software engineering:
   - Leverages OpenHands' battle-tested sandbox and code execution
   - Required for SWEBench evaluation (file editing, git patches, test execution)
   - Integrates with OpenHands' rich tooling ecosystem

### Trajectory Wrappers

Each agent has a corresponding trajectory wrapper:

| Wrapper | Agent | Location |
|---------|-------|----------|
| `ReActTrajectory` | `ReActAgent` | `agents/react/react_runner.py:8` |
| `CodeActTrajectory` | `OHCodeActAgent` | `agents/oh_codeact/codeact_runner.py:98` |

**ReActTrajectory** is simple:
```python
async def generate_trajectory(self):
    self.agent = ReActAgent(...)
    instruction = self.task.get_instruction(instance)
    finish_reason, result = await self.agent.run(instruction, instance)
    # Collect messages and transitions
```

**CodeActTrajectory** manages the OpenHands lifecycle:
```python
async def initialize_trajectory(self):
    runtime = await self.task.initialize_runtime(...)  # Start sandbox
    self.agent.runtime = runtime

async def generate_trajectory(self):
    state = await run_controller(runtime=runtime, agent=agent, ...)  # OpenHands loop
    result = await self.task.complete_runtime(runtime, ...)  # Extract git patch

def _cleanup_agent(self):
    self.agent.close()  # Cleanup runtime
```

## Key Data Structures

| Class | Location | Purpose |
|-------|----------|---------|
| `Transition` | `functional/utils.py:29` | Single LLM call record: `ob`, `ac`, `reward`, `episode_done` |
| `MessageHistory` | `functional/history.py:16` | Conversation state management |
| `MessageEncoder` | `functional/history.py:99` | Message → token_ids with incremental encoding |
| `StepResult` | `functional/utils.py` | Wrapper for step outputs |

## Separation of Concerns

**Loosely coupled:**
- **Agents** are decoupled from backends via `AsyncInferBackend` interface
- **Tasks** are decoupled from agents via `BaseTask` interface
- **Tools** are registered in `TOOL_REGISTRY` and called generically

**Tightly coupled:**
- `ReActTrajectory` ↔ `ReActAgent`: trajectory wrapper directly instantiates and calls the agent
- `SkyAgentLoopManager` ↔ VeRL internals: tightly integrates with VeRL's DataProto format

## VeRL Integration

VeRL-specific code lives under `integrations/verl/`. The integration points:

```
┌─────────────────────────────────────────────────────────────────────┐
│  verl_main_ppo.py                                                   │
│  - Entry point for VeRL PPO training                                │
│  - SkyAgentRewardManager: computes rewards                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SkyAgentPPOTrainer (verl_trainer.py:61)                            │
│  - Extends VeRL's RayPPOTrainer                                     │
│  - Distributes training across Ray workers                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SkyAgentLoopManager (verl_async_manager.py:60)                     │
│  - Rollout worker: runs agent trajectories during training          │
│  - generate_sequences(): instantiates AgentRunner, runs rollouts    │
│  - _postprocess(): converts output → DataProto for RL training      │
│    (pads prompts/responses, creates attention masks, position IDs)  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VeRLBackend (verl_backend.py:6)                                    │
│  - Implements AsyncInferBackend for VeRL inference                  │
│  - VeRLGeneratorInput/Output: tensor wrappers for VeRL data format  │
└─────────────────────────────────────────────────────────────────────┘
```

### What's VeRL-Specific vs General

| VeRL-Specific | General (Backend-Agnostic) |
|---------------|----------------------------|
| `SkyAgentPPOTrainer` | `AgentRunner` |
| `SkyAgentLoopManager` | `BaseTrajectory` / `ReActTrajectory` |
| `SkyAgentRewardManager` | `ReActAgent` |
| `VeRLBackend`, `VeRLGeneratorInput/Output` | `AsyncInferBackend` interface |
| DataProto padding/masking in `_postprocess()` | `Transition`, `MessageHistory` |

The core agent logic (`ReActAgent.step()`, `MessageHistory`, `Transition`) is fully backend-agnostic. VeRL integration wraps this core with:
1. Distributed training coordination (`SkyAgentPPOTrainer`)
2. Rollout management (`SkyAgentLoopManager`)
3. Data format conversion (padding, masking for PPO)

## Environment Interaction

Agents interact with environments via **Tasks** and **Tools**:

```
┌─────────────────────┐      ┌─────────────────────┐
│      BaseTask       │      │    TOOL_REGISTRY    │
│  (tasks/base.py)    │      │   (tools/base.py)   │
├─────────────────────┤      ├─────────────────────┤
│ get_instruction()   │      │ finish              │
│ initialize_runtime()│      │ sandbox_fusion      │
│ evaluate_result()   │      │ web_browser         │
│ complete_runtime()  │      │ search_engine       │
└─────────────────────┘      └─────────────────────┘
         │                            │
         │ provides initial           │ agent calls tools
         │ messages & evaluation      │ during step()
         ▼                            ▼
┌─────────────────────────────────────────────────────┐
│                    ReActAgent                       │
│  - Receives task instruction as initial messages    │
│  - Calls tools via TOOL_REGISTRY[name].call()       │
│  - Tools can access agent state via agent= param    │
└─────────────────────────────────────────────────────┘
```

**Task implementations:**
- `general_react/` - General ReAct tasks
- `swebench/` - Software engineering benchmark
- `web_research_task.py` - Web research tasks

**Tool interface:** Each tool receives `(args, agent=self)`, allowing access to `agent.instance`, `agent.history`, etc.

## Entry Points

| Use Case | Entry Point | Config |
|----------|-------------|--------|
| Standalone (OpenAI) | `examples/run_openai/` | Task YAML |
| VeRL PPO training | `integrations/verl/verl_main_ppo.py` | Hydra |
| Tinker inference | `integrations/tinker/tinker_train.py` | Hydra |
| skyrl-train | `integrations/skyrl_train/skyrl_train_main.py` | Hydra |

## Key Files for Deep Dive

1. **Agent loop core:** `agents/react/react_agent.py:311` (`step()` method)
2. **Trajectory wrapper:** `agents/react/react_runner.py:8` (`ReActTrajectory`)
3. **Batch orchestration:** `agents/base.py:629` (`AgentRunner.run()`)
4. **VeRL rollout:** `integrations/verl/verl_async_manager.py:209` (`generate_sequences()`)
5. **Message encoding:** `functional/history.py:99` (`MessageEncoder`)
6. **Transition recording:** `functional/utils.py:29` (`Transition` dataclass)
