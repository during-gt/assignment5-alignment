from .checkpoint import tokenize_prompt_and_output, compute_rollout_rewards, compute_group_normalized_rewards, get_response_log_probs, compute_policy_gradient_loss, aggregate_loss_across_microbatch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer
from typing import Callable, Literal
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
        mb_old_log_probs = old_log_probs[start:end] if old_log_probs is not None else None

        mb_log_probs = get_response_log_probs(model, mb_input_ids, mb_labels)["log_probs"]

        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            mb_advantages, mb_log_probs, importance_reweighting_method,
            mb_old_log_probs, cliprange, mb_response_mask,
        )

        loss = aggregate_loss_across_microbatch(
            per_token_loss, mb_response_mask, loss_normalization, normalization_constant
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss = total_loss + loss.detach()

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        **reward_metadata,
        **adv_metadata,
        **loss_metadata,
        "grad_norm": grad_norm.item() if grad_norm is not None else None,
    }
    return total_loss, metadata


