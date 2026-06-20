import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer,PreTrainedModel

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
    prompt_and_output_strs = [p + o for p, o in zip(prompt_strs, output_strs)]
    tokenized = tokenizer(
        prompt_and_output_strs,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[:, :-1]
    labels = tokenized.input_ids[:, 1:]
    response_mask = torch.zeros_like(labels)
    for i in range(batch_size):
        prompt_len = len(tokenizer(prompt_strs[i]).input_ids)
        real_len = tokenized.attention_mask[i].sum().item()  # 不含 padding 的真实 token 数
        response_mask[i, prompt_len - 1 : real_len - 1] = 1
    return {    
        "input_ids": input_ids, # input_ids: [p1, p2, p3, r1, r2]     ← 送入模型做 forward
        "labels": labels,       # labels:    [p2, p3, r1, r2, r3]     ← 每个位置的"正确下一个 token"
        "response_mask": response_mask, # resp_mask: [ 0,  0,  1,  1,  1]     ← 1 的位置 = response token
    }


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor,return_token_entropy: bool = False,) -> dict[str, torch.Tensor]:
    """
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
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = log_probs.shape
        log_probs = log_probs.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        token_log_probs = log_probs[torch.arange(batch_size * seq_len), labels_flat].view(batch_size, seq_len)
        result = {"log_probs": token_log_probs}
        
        if return_token_entropy:
            probs = torch.exp(log_probs)
            token_entropy = -torch.sum(probs * log_probs, dim=-1).view(batch_size, seq_len)
            result["token_entropy"] = token_entropy
        return result
    


def compute_rollout_rewards(reward_fn: Callable[[str, str], dict[str, float]], rollout_responses: list[str], repeated_ground_truths: list[str],) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Args:
        • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        • rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        • repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
    
    Returns:
        • tuple[torch.Tensor, dict[str, float]].
        ‣ raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        ‣ metadata Reward statistics to log. At minimum, include the mean total and format rewards over the rollout batch.
    """