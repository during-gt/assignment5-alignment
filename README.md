# CS336 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)


## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```


## GRPO training (GSM8K)

### Dependencies for logging

Experiment logging uses [Weights & Biases](https://wandb.ai). `wandb` is already
declared in `pyproject.toml` as an optional dependency (the `gpu` and `plots`
extras), so it is *not* installed by a plain `uv sync`. Install it (and log in
once) with:

```sh
uv sync --extra plots      # or: uv sync --extra gpu
uv run wandb login         # one-time; paste the API key from https://wandb.ai/authorize
```

`wandb` is optional: if it is not installed, training falls back to printing all
metrics to stdout/the log file, so you can skip it entirely if you only want the
text logs.

### Run standard on-policy GRPO

Trains `allenai/OLMo-2-0425-1B` on GSM8K. The policy trains on `cuda:0` and a
vLLM server (started automatically) serves rollouts on the second GPU.

```sh
mkdir -p logs
nohup uv run python -m cs336_alignment.grpo_training \
    --num-rollout-steps 50 \
    --eval-interval 10 \
    --log-rollouts-interval 10 \
    > logs/grpo_50steps.log 2>&1 &
echo "PID: $!"
```

Useful flags (see `--help` for all of them):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-rollout-steps` | `50` | Number of rollout/optimizer steps |
| `--eval-interval` | `10` | Evaluate on the validation set every N steps |
| `--log-rollouts-interval` | `10` | Print sample training rollouts every N steps |
| `--n-val-examples` | `1024` | Validation examples per eval (keep >= 1024; eval is noisy) |
| `--device` | `cuda:0` | Device for the training policy |
| `--learning-rate` | `1e-5` | AdamW learning rate |
| `--model-name` | `allenai/OLMo-2-0425-1B` | HF model to train |
| `--train-file` / `--val-file` | `data/gsm8k/{train,test}.jsonl` | Datasets |

### Monitor progress

Validation reward should improve and rollouts should get cleaner over the run.

```sh
tail -f logs/grpo_50steps.log                                        # live
grep -E "val/answer_accuracy|val/format_rate" logs/grpo_50steps.log  # val reward trend
grep -A5 "sample training rollouts" logs/grpo_50steps.log            # qualitative rollouts
grep "View run" logs/grpo_50steps.log | tail -1                      # wandb URL for this run
```

Metrics are logged at three cadences: `train/*` every step, while `val/*` and the
sample rollouts are logged every `--eval-interval` / `--log-rollouts-interval`
steps. To stop a backgrounded run:

```sh
pkill -f "cs336_alignment.grpo_training" ; pkill -f "vllm serve"
```

