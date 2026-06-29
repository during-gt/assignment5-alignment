from .checkpoint import tokenize_prompt_and_output, compute_rollout_rewards, compute_group_normalized_rewards, get_response_log_probs, compute_policy_gradient_loss, aggregate_loss_across_microbatch, get_model_and_tokenizer
from .drgrpo_grader import r1_zero_reward_fn
from .vllm_utils import VLLMServer
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer
from typing import Callable, Literal
import json
import os
import random
import torch
def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    Execute forward-and-backward passes, with gradient_accumulation_steps microbatches

    only need to implement this function for standard GRPO in the on-policy setting, or baseline = "mean", advantage_normalizer = "std",
    importance_reweighting_method = "none", and loss_normalization = "sequence". Feel free to
    raise a NotImplementedError for unsupported inputs

    Args:
        • model: PreTrainedModel HuggingFace model to train.
        • tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
        • optimizer: Optimizer Optimizer for the model.
        • gradient_accumulation_steps: int Number of microbatches per optimizer step.
        • max_grad_norm: float | None If not None, clip the gradient norm to this value before calling optimizer.step().
        • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        • repeated_prompts: list[str] The prompts for the examples. The length of this list is rollout_batch_size, because the prompt for each example is repeated group_size times.
        • rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        • repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
        • group_size: int Number of responses per question (group).
        • baseline: Literal["mean", "none"] If mean, subtract the per-group mean reward; if none, do nothing.
        • advantage_eps: float Small constant to avoid division by zero in normalization.
        • advantage_normalizer: Literal["std", "none", "mean"] If std, divide by the per-group standard deviation; if none, do nothing; if mean, divide by the per-group mean reward.
        • importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] "none": no importance reweighting; "noclip": apply importance reweighting without clipping; "grpo": do
        PPO/GRPO-style token-level reweighting and clipping; "gspo": do GSPO-style sequence-level reweighting and clipping.
        • old_log_probs: torch.Tensor | None Required unless importance_reweighting_method = "none"; shape (batch_size, sequence_length).
        • cliprange: float | None = None Clip parameter 𝜀, required when importance_reweighting_method is "grpo" or "gspo".
        • loss_normalization: Literal["sequence", "constant"] = "sequence" "sequence": average loss over each sequence, then average over sequences; "constant": normalize total loss by a constant
        (fixed for all of training).
        • normalization_constant: int | None = None The constant to divide total loss by; required if loss_normalization = "constant".
    
    Returns:
        • tuple[torch.Tensor, dict[str, torch.Tensor]].
            ‣ loss scalar tensor. The batch loss, adjusted for gradient accumulation. We return this so we can log it.
            ‣ metadata Dict with metadata from the underlying loss call, gradient norm before clipping, and any other statistics you might want to log.
    """
    if baseline != "mean":
        raise NotImplementedError(f"baseline={baseline!r} is not supported; only 'mean' is implemented")
    if advantage_normalizer != "std":
        raise NotImplementedError(f"advantage_normalizer={advantage_normalizer!r} is not supported; only 'std' is implemented")
    if importance_reweighting_method != "none":
        raise NotImplementedError(f"importance_reweighting_method={importance_reweighting_method!r} is not supported; only 'none' is implemented")
    if loss_normalization != "sequence":
        raise NotImplementedError(f"loss_normalization={loss_normalization!r} is not supported; only 'sequence' is implemented")

    # tokenize
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)

    # rewards and advantages don't need gradients
    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, adv_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    advantages = advantages.to(device)

    # gradient accumulation over microbatches
    # rollout batch 可能有几百条序列、每条最多 512 token,整批一次性 forward+backward 显存装不下。于是切成小块、逐块 backward() 累积梯度,数学上等价于一次性大 batch,但峰值显存只需 1/gradient_accumulation_steps。
    rollout_batch_size = input_ids.shape[0]
    microbatch_size = rollout_batch_size // gradient_accumulation_steps

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    for i in range(gradient_accumulation_steps):
        start = i * microbatch_size
        end = start + microbatch_size

        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_response_mask = response_mask[start:end]
        mb_advantages = advantages[start:end]

        mb_log_probs = get_response_log_probs(model, mb_input_ids, mb_labels)["log_probs"]

        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            mb_advantages, mb_log_probs,
        )

        loss = aggregate_loss_across_microbatch(
            per_token_loss, mb_response_mask, loss_normalization, normalization_constant
        )
        loss = loss / gradient_accumulation_steps # scale down loss for gradient accumulation
        loss.backward() # 累积梯度,但不更新!
        total_loss = total_loss + loss.detach()

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step() #update parameters
    optimizer.zero_grad() # zero out gradients for next step

    metadata = {
        **reward_metadata,
        **adv_metadata,
        **loss_metadata,
        "grad_norm": grad_norm.item() if grad_norm is not None else None,
    }
    return total_loss, metadata

#=========full training loop for GRPO==========#
# Default hyperparameters for standard on-policy GRPO. Any of these may be
# overridden through the `training_hyperparameters` / `sampling_hyperparameters`
# dicts passed to grpo_experiments_standard_on_policy.
DEFAULT_TRAINING_HYPERPARAMETERS = {
    "n_val_examples": 1024,            # evaluate on at least this many val examples
    "num_rollout_steps": 200,          # number of rollout/optimizer steps
    "learning_rate": 1e-5,
    "rollout_batch_size": 256,         # rollouts per step = n_prompts_per_rollout_batch * group_size
    "group_size": 8,                   # responses sampled per prompt (group)
    "gradient_accumulation_steps": 32,
    "max_grad_norm": 1.0,
    "eval_interval": 10,               # evaluate on val set every N rollout steps
    "log_rollouts_interval": 40,       # log training rollouts every N rollout steps
    "seed": 42,
    # GRPO advantage / loss configuration (standard on-policy defaults)
    "baseline": "mean",
    "advantage_eps": 1e-6,
    "advantage_normalizer": "std",
    "loss_normalization": "sequence",
    "device": "cuda:0",
}

DEFAULT_SAMPLING_HYPERPARAMETERS = {
    "temperature": 1.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
    "eval_batch_size": 64,
}

# Prompt template used to turn a raw GSM8K question into a policy prompt.
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
R1_ZERO_PROMPT_PATH = os.path.join(PROMPT_DIR, "r1_zero.prompt")


def _load_gsm8k(path: str) -> tuple[list[str], list[str]]:
    """Load a GSM8K jsonl file, returning (questions, ground_truth_answers)."""
    questions, ground_truths = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append(item["question"])
            # Ground truth is the text after the final "#### " marker.
            ground_truths.append(item["answer"].split("#### ")[-1].strip())
    return questions, ground_truths


def _evaluate_on_validation(
    vllm_server: VLLMServer,
    template: str,
    questions: list[str],
    ground_truths: list[str],
    sampling_params: dict,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_batch_size: int,
) -> tuple[dict[str, float], list[dict]]:
    """Greedy-ish single-sample evaluation over the validation set.

    Returns (metrics, records) where metrics holds mean answer/format/total
    reward and records holds per-example generations for qualitative logging.
    """
    prompts = [template.format(question=q) for q in questions]
    eval_sampling = dict(sampling_params)
    eval_sampling["n"] = 1
    eval_sampling.setdefault("seed", 0)  # generate_completions requires a seed
    completions = vllm_server.generate_completions(prompts, eval_sampling, batch_size=eval_batch_size)

    records = []
    answer_total = format_total = reward_total = 0.0
    for q, comp, gt in zip(questions, completions, ground_truths):
        reward = reward_fn(comp.text, gt)
        answer_total += reward["answer_reward"]
        format_total += reward["format_reward"]
        reward_total += reward["reward"]
        records.append({
            "question": q,
            "generation": comp.text,
            "ground_truth": gt,
            "answer_reward": reward["answer_reward"],
            "format_reward": reward["format_reward"],
            "reward": reward["reward"],
        })

    n = len(records)
    metrics = {
        "val/answer_accuracy": answer_total / n,
        "val/format_rate": format_total / n,
        "val/mean_reward": reward_total / n,
        "val/n_examples": n,
    }
    return metrics, records


def grpo_experiments_standard_on_policy(model_name: str, training_set_file_path: str, validation_set_file_path: str, sampling_hyperparameters: dict, training_hyperparameters: dict):
    """Run standard on-policy GRPO training on GSM8K.

    Each rollout step: sync the policy weights into the vLLM server, sample a
    batch of prompts, generate `group_size` responses per prompt, take a single
    GRPO optimizer step over those rollouts, and periodically evaluate on the
    validation set (logging some generations for a qualitative sense of progress).
    """
    # =======initialize hyperparameters========
    hp = {**DEFAULT_TRAINING_HYPERPARAMETERS, **(training_hyperparameters or {})}
    sp = {**DEFAULT_SAMPLING_HYPERPARAMETERS, **(sampling_hyperparameters or {})}

    group_size = hp["group_size"]
    rollout_batch_size = hp["rollout_batch_size"]
    if rollout_batch_size % group_size != 0:
        raise ValueError(f"rollout_batch_size ({rollout_batch_size}) must be divisible by group_size ({group_size})")
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    device = hp["device"]
    rng = random.Random(hp["seed"])

    # =======Load datasets and prompt template======
    train_questions, train_ground_truths = _load_gsm8k(training_set_file_path)
    val_questions, val_ground_truths = _load_gsm8k(validation_set_file_path)
    # Make sure we evaluate on at least n_val_examples (CoT/RL eval is noisy).
    n_val = min(hp["n_val_examples"], len(val_questions))
    n_val = max(n_val, min(1024, len(val_questions)))
    val_questions = val_questions[:n_val]
    val_ground_truths = val_ground_truths[:n_val]

    with open(R1_ZERO_PROMPT_PATH) as f:
        template = f.read()

    reward_fn = r1_zero_reward_fn

    # =======Load model and optimizer========
    policy, tokenizer = get_model_and_tokenizer(model_name, device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=hp["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0
    )

    # =======initialize the vLLM server and NCCL weight sync========
    vllm_server = VLLMServer(model_id=model_name, gpu_memory_utilization=0.4)
    vllm_server.start()
    vllm_server.init_weight_sync(device)

    # =======initialize wandb logging (optional)========
    try:
        import wandb
        wandb.init(project="cs336-grpo", config={**hp, **sp, "model_name": model_name})
        use_wandb = True
    except Exception:
        wandb = None
        use_wandb = False

    def log_metrics(metrics: dict, step: int):
        if use_wandb:
            wandb.log(metrics, step=step)
        printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
        print(f"[step {step}] {printable}")

    # Sampling params used to generate training rollouts (group_size samples/prompt).
    rollout_sampling = {
        "temperature": sp["temperature"],
        "max_tokens": sp["max_tokens"],
        "top_p": sp["top_p"],
        "n": group_size,
        "stop": sp["stop"],
        "include_stop_str_in_output": sp["include_stop_str_in_output"],
    }

    # ======= training loop ========
    train_indices = list(range(len(train_questions)))
    cursor = 0
    for step in range(hp["num_rollout_steps"]):
        # ---- sync the latest policy weights into the vLLM server ----
        vllm_server.sync_policy_weights(policy)

        # ---- sample a batch of unique prompts (reshuffle each epoch) ----
        if cursor + n_prompts_per_rollout_batch > len(train_indices):
            rng.shuffle(train_indices)
            cursor = 0
        batch_idx = train_indices[cursor:cursor + n_prompts_per_rollout_batch]
        cursor += n_prompts_per_rollout_batch

        batch_questions = [train_questions[i] for i in batch_idx]
        batch_ground_truths = [train_ground_truths[i] for i in batch_idx]
        prompts = [template.format(question=q) for q in batch_questions]

        # ---- produce training rollouts (group_size responses per prompt) ----
        rollout_sampling_step = dict(rollout_sampling)
        rollout_sampling_step["seed"] = hp["seed"] + step
        completions = vllm_server.generate_completions(
            prompts, rollout_sampling_step, batch_size=n_prompts_per_rollout_batch
        )
        # vLLM returns group_size contiguous completions per prompt (sorted by
        # index), so repeating prompts/ground-truths group_size times aligns the
        # rollouts with their group for group-normalized advantages.
        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(group_size)]
        rollout_responses = [c.text for c in completions]

        # ---- take a GRPO policy-gradient step over the rollouts ----
        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=hp["gradient_accumulation_steps"],
            max_grad_norm=hp["max_grad_norm"],
            reward_fn=reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            baseline=hp["baseline"],
            advantage_eps=hp["advantage_eps"],
            advantage_normalizer=hp["advantage_normalizer"],
            loss_normalization=hp["loss_normalization"],
        )

        metrics = {"train/loss": loss.item()}
        for k, v in metadata.items():
            if v is not None:
                metrics[f"train/{k}"] = v
        log_metrics(metrics, step)

        # ---- periodically log training rollouts for a qualitative sense ----
        if step % hp["log_rollouts_interval"] == 0:
            sample = []
            for j in range(min(4, len(rollout_responses))):
                sample.append({
                    "prompt": repeated_prompts[j],
                    "response": rollout_responses[j],
                    "ground_truth": repeated_ground_truths[j],
                })
            print(f"[step {step}] sample training rollouts:")
            for s in sample:
                print(f"  GT={s['ground_truth']!r} :: {s['response'].strip()[:300]!r}")
            if use_wandb:
                table = wandb.Table(columns=["prompt", "response", "ground_truth"])
                for s in sample:
                    table.add_data(s["prompt"], s["response"], s["ground_truth"])
                wandb.log({"train/rollouts": table}, step=step)

        # ---- periodically evaluate on the validation set ----
        if step % hp["eval_interval"] == 0 or step == hp["num_rollout_steps"] - 1:
            policy.eval()
            vllm_server.sync_policy_weights(policy)
            with torch.no_grad():
                val_metrics, val_records = _evaluate_on_validation(
                    vllm_server, template, val_questions, val_ground_truths,
                    sampling_params=rollout_sampling, reward_fn=reward_fn,
                    eval_batch_size=sp["eval_batch_size"],
                )
            log_metrics(val_metrics, step)
            if use_wandb:
                table = wandb.Table(columns=["question", "generation", "ground_truth", "reward"])
                for r in val_records[:8]:
                    table.add_data(r["question"], r["generation"], r["ground_truth"], r["reward"])
                wandb.log({"val/generations": table}, step=step)
            policy.train()

    if use_wandb:
        wandb.finish()
    vllm_server.stop()
    return policy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standard on-policy GRPO training on GSM8K.")
    parser.add_argument("--model-name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-file", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-file", default="data/gsm8k/test.jsonl")
    parser.add_argument("--num-rollout-steps", type=int, default=50,
                        help="Number of rollout/optimizer steps to run.")
    parser.add_argument("--n-val-examples", type=int, default=1024,
                        help="Number of validation examples to evaluate on (>=1024 recommended).")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate on the validation set every N steps.")
    parser.add_argument("--log-rollouts-interval", type=int, default=10,
                        help="Print sample training rollouts every N steps.")
    parser.add_argument("--device", default="cuda:0", help="Device for the training policy.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    training_hyperparameters = {
        "num_rollout_steps": args.num_rollout_steps,
        "n_val_examples": args.n_val_examples,
        "eval_interval": args.eval_interval,
        "log_rollouts_interval": args.log_rollouts_interval,
        "device": args.device,
        "learning_rate": args.learning_rate,
    }
    sampling_hyperparameters = {}

    grpo_experiments_standard_on_policy(
        model_name=args.model_name,
        training_set_file_path=args.train_file,
        validation_set_file_path=args.val_file,
        sampling_hyperparameters=sampling_hyperparameters,
        training_hyperparameters=training_hyperparameters,
    )


if __name__ == "__main__":
    main()
