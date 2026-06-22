import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer,PreTrainedModel
from typing import Callable
from typing_extensions import Literal

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer,) -> dict[str, torch.Tensor]:
    """
    Args:
        • prompt_strs: list[str] List of repeated prompt strings in a batch. ["Hello", "Hello", "This is", "This is"]
        • output_strs: list[str] List of rollout output strings in a batch.["world", "test",  "a test",  "another test"]
        • tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the concatenated tokenized prompt and output strings.
        Then the returned dictionary should havethe following keys:
            ‣ input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the
            tokenized prompt and output strings, with the final token sliced off.
            ‣ labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input
            ids, i.e., the input ids without the first token.
            ‣ response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask aligned with labels, with value 1 where the corresponding label token is part of the
            response and 0 otherwise.
    """
    batch_size = len(prompt_strs)
    prompt_ids_list = [tokenizer.encode(p) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o) for o in output_strs]
    combined_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    prompt_lens = [len(p) for p in prompt_ids_list]
    total_lens = [len(c) for c in combined_ids_list]

    max_len = max(total_lens)
    pad_id = tokenizer.pad_token_id
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in combined_ids_list]
    padded_tensor = torch.tensor(padded)

    input_ids = padded_tensor[:, :-1]
    labels = padded_tensor[:, 1:]
    response_mask = torch.zeros_like(labels)
    for i in range(batch_size):
        pl = prompt_lens[i]
        tl = total_lens[i]
        response_mask[i, pl - 1 : tl - 1] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    # batch_size = len(prompt_strs)
    # prompt_and_output_strs = [p + o for p, o in zip(prompt_strs, output_strs)]
    # tokenized = tokenizer(
    #     prompt_and_output_strs,
    #     padding="longest",
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # input_ids = tokenized.input_ids[:, :-1]
    # labels = tokenized.input_ids[:, 1:]
    # response_mask = torch.zeros_like(labels)
    # for i in range(batch_size):
    #     prompt_len = len(tokenizer(prompt_strs[i]).input_ids)
    #     real_len = tokenized.attention_mask[i].sum().item()  # 不含 padding 的真实 token 数
    #     response_mask[i, prompt_len - 1 : real_len - 1] = 1
    # return {    
    #     "input_ids": input_ids, # input_ids: [p1, p2, p3, r1, r2]     ← 送入模型做 forward
    #     "labels": labels,       # labels:    [p2, p3, r1, r2, r3]     ← 每个位置的"正确下一个 token"
    #     "response_mask": response_mask, # resp_mask: [ 0,  0,  1,  1,  1]     ← 1 的位置 = response token
    # }


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor,return_token_entropy: bool = False,) -> dict[str, torch.Tensor]:
    """
    Gets per-token conditional logprobabilities (given the previous tokens) from a causal language model, and optionally the entropy of the model’s next-token distribution
    
    Args:
        • model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).
        • input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.
        • labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method.
        • return_token_entropy: bool If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            ‣ "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log 𝑝𝜃(𝑥𝑡 | 𝑥<𝑡).
            ‣ "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True).

    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = log_probs.shape
    log_probs = log_probs.view(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    token_log_probs = log_probs[torch.arange(batch_size * seq_len), labels_flat].view(batch_size, seq_len)
    result = {"log_probs": token_log_probs}
    
    if return_token_entropy:
        probs = torch.exp(log_probs)
        token_entropy = -torch.sum(probs * log_probs, dim=-1).view(batch_size, seq_len)
        result["token_entropy"] = token_entropy
    return result
    


def compute_rollout_rewards(reward_fn: Callable[[str, str], dict[str, float]], rollout_responses: list[str], repeated_ground_truths: list[str],) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calculates raw rewards for each rollout response.

    Args:
        • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        • rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        • repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
    
    Returns:
        • tuple[torch.Tensor, dict[str, float]].
        ‣ raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        ‣ metadata Reward statistics to log. At minimum, include the mean total and format rewards over the rollout batch.
    """
    raw_rewards = []
    total_reward = 0.0
    format_reward_total = 0.0
    answer_reward_total = 0.0
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        rewards = reward_fn(response, gt)
        raw_rewards.append(rewards["reward"])
        total_reward += rewards["reward"]
        format_reward_total += rewards["format_reward"]
        answer_reward_total += rewards["answer_reward"]
    
    metadata = {
        "mean_total_reward": total_reward / len(rollout_responses),
        "mean_format_reward": format_reward_total / len(rollout_responses),
        "mean_answer_reward": answer_reward_total / len(rollout_responses),
    }
    return torch.tensor(raw_rewards), metadata  

def compute_group_normalized_rewards(raw_rewards: torch.Tensor, group_size: int, baseline: Literal["mean", "none"] = "mean", advantage_eps: float = 1e-6, advantage_normalizer: Literal["std", "none", "mean"] = "std",):
    """
    Args:
        • raw_rewards: torch.Tensor shape (rollout_batch_size,). Unnormalized rewards for each rollout response, where rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        • group_size: int Number of responses per question (group).
        • baseline: Literal["mean", "none"] For this problem, support mean, which subtracts the pergroup mean reward. Later, none will mean no baseline subtraction.
        • advantage_eps: float Small constant to avoid division by zero in normalization.
        • advantage_normalizer: Literal["std", "none", "mean"] For this problem, support std, which divides by the per-group standard deviation. Later, none will mean no normalization and mean will mean divide by the per-group mean reward.
    
    Returns:
        tuple[torch.Tensor, dict[str, float]].
            ‣ advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            ‣ metadata your choice of other statistics to log (e.g.mean, std, max/min of rewards).
    """
    rollout_batch_size = raw_rewards.shape[0]
    n_groups = rollout_batch_size // group_size
    advantages = torch.zeros_like(raw_rewards)
    
    for i in range(n_groups):
        group_rewards = raw_rewards[i * group_size : (i + 1) * group_size]
        if baseline == "mean":
            group_baseline = group_rewards.mean()
        else:
            group_baseline = 0.0
        
        if advantage_normalizer == "std":
            normalizer = group_rewards.std() + advantage_eps
        elif advantage_normalizer == "mean":
            normalizer = group_rewards.mean() + advantage_eps
        else:
            normalizer = 1.0
        
        advantages[i * group_size : (i + 1) * group_size] = (group_rewards - group_baseline) / normalizer
    
    metadata = {
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
        "max_advantage": advantages.max().item(),
        "min_advantage": advantages.min().item(),
    }
    return advantages, metadata

def compute_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, 
                                policy_log_probs: torch.Tensor,  importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
                                old_log_probs: torch.Tensor | None = None, 
                                cliprange: float | None = None,
                                response_mask: torch.Tensor | None = None,) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.
    Args:
        • raw_rewards_or_advantages: torch.Tensor Shape (rollout_batch_size,) or (rollout_batch_size, 1), scalar reward/advantage for each rollout response.
        • policy_log_probs: torch.Tensor Shape (rollout_batch_size, sequence_length), logprobs for each token.
        • importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] "none": no importance reweighting; "noclip": apply importance reweighting without clipping; "grpo": do PPO/GRPO-style token-level reweighting and clipping; "gspo": do GSPO-style sequence-level reweighting and clipping.
        • old_log_probs: torch.Tensor | None Required unless importance_reweighting_method = "none"; shape (batch_size, sequence_length).
        • cliprange: float | None = None Clip parameter 𝜀, required when importance_reweighting_method is "grpo" or "gspo".
        • response_mask: torch.Tensor | None = None Optional shape (batch_size, sequence_length) mask over response tokens. Required for GSPO implementations that average the sequencelevel log-ratio over response tokens only.
    
    Returns:
        • loss: torch.Tensor A scalar containing the average loss. Make sure you can later call backward on this loss.
        • metatdata: dict[str, torch.Tensor] A dictionary containing statistics you want to log
    """
    # reshape to (rollout_batch_size, 1) so it broadcasts across sequence length
    advantages = raw_rewards_or_advantages.view(-1, 1)
    if importance_reweighting_method == "none":
        loss = -(advantages * policy_log_probs)  # (batch_size, seq_length)

    metadata = {"mean_loss": loss.mean().item()}
    return loss, metadata


def aggregate_loss_across_microbatch(per_token_policy_gradient_loss: torch.Tensor, mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor:
    """
    Aggregate the per-token policy-gradient loss according to the response mask and lossnormalization strategy.

    Args:
        • per_token_policy_gradient_loss: torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss 
        (to be aggregated across the batch and sequence dimensions in the training loop).
        • mask torch.Tensor of shape (batch_size, sequence_length) denoting which positions should be included in the loss.
        • loss_normalization: Literal["sequence", "constant"] = "sequence" "sequence": average loss over each sequence, 
        then average over sequences; "constant": normalize total loss by a constant.
        • normalization_constant: int | None = None The constant to divide total loss by; required if loss_normalization = "constant".

    Returns:
        • loss: torch.Tensor A scalar containing the average loss. Make sure you can later call backward on this loss.
    """
    if loss_normalization == "sequence":
        loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # average over response/sequence length
        loss = loss.mean()  # average over batch size and  group_size 
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant must be provided when loss_normalization is 'constant'")
        loss = (per_token_policy_gradient_loss * mask).sum() / normalization_constant
    else:
        raise ValueError(f"Unsupported loss_normalization: {loss_normalization}")
    return loss
