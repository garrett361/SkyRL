# SkyRL Learning Context

## Purpose
This branch is for learning SkyRL internals. Responses should be educational and precise.

## Response Guidelines

### For Architecture Questions
- Always provide ASCII diagrams showing component relationships
- Include data flow arrows and key interfaces
- Label with relevant file paths

### For All Other Questions
- Provide concise, targeted explanations
- Include file paths with line numbers: `skyrl-train/skyrl/trainer.py:42`
- Reference specific classes/functions by name
- Keep responses focused—avoid boilerplate

## Repository Structure

```
SkyRL/
├── skyrl-train/   # Core RL training (PPO, GRPO, REINFORCE)
├── skyrl-gym/     # Gymnasium environments for LLMs
├── skyrl-agent/   # Multi-turn agent framework
├── skyrl-tx/      # Tinker API backend (JAX/Flax)
└── docs/          # Documentation site
```

## Key Entry Points

| Component | Main Entry | Config |
|-----------|------------|--------|
| Training | `skyrl-train/skyrl/trainer.py` | Hydra (`config/`) |
| Environments | `skyrl-gym/skyrl_gym/core.py` | Python dataclasses |
| Agents | `skyrl-agent/skyrl_agent/agents/` | Hydra |
| Inference | `skyrl-tx/skyrl_tx/tinker/` | FastAPI |

## Tech Stack
- **Training**: PyTorch, Ray, Megatron-LM, FSDP
- **Inference**: vLLM, SGLang, Tinker API
- **Config**: Hydra/OmegaConf
- **Tracking**: Weights & Biases

## Learning Notes

Previous exploration summaries are stored under `learn_skyrl/`. Check these before exploring a component:

| Component | Summary Location |
|-----------|------------------|
| skyrl-agent | `learn_skyrl/skyrl-agent/README.md` |
