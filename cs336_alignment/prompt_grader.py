import os
import json

import cs336_alignment.vllm_utils as vllm_utils
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# Define vLLM server parameters
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
model_id = "allenai/OLMo-2-0425-1B"

sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "n": 1,
    "seed": 42,
}

#Read Prompts
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_FILES = {
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
    "question_only": "question_only.prompt",
}


class prompt_grader:
    """
    Evaluate allenai/OLMo-2-0425-1B on GSM8K with zero-shot question_only,
    zero-shot r1_zero, and few-shot r1_zero_three_shot prompts.
    """

    def __init__(self, prompt_type: str):
        self.prompt_type = prompt_type
        self.sampling_params = dict(sampling_params) #shallow copy 
        if prompt_type in ("r1_zero", "r1_zero_three_shot"):
            self.sampling_params["stop"] = ["</answer>"]
            self.sampling_params["include_stop_str_in_output"] = True

        # Load the prompt template
        with open(os.path.join(PROMPT_DIR, PROMPT_FILES[prompt_type])) as f:
            self.template = f.read()

        # Build and start the server (start() must be called before generating).
        self.vllm_server = vllm_utils.VLLMServer(model_id=model_id, host=VLLM_HOST, port=VLLM_PORT, gpu_memory_utilization=0.4)
        self.vllm_server.start()


    def grade_all(self, questions: list[str], ground_truths: list[str], batch_size: int = 64) -> list[dict]:
        """Grade a batch of questions; returns one record per question with the generation, ground truth, and the three reward components."""
        prompts = [self.template.format(question=q) for q in questions]
        completions = self.vllm_server.generate_completions(prompts, self.sampling_params, batch_size=batch_size)
        reward_fn = question_only_reward_fn if self.prompt_type == "question_only" else r1_zero_reward_fn
        records = []
        for q, comp, gt in zip(questions, completions, ground_truths):
            reward = reward_fn(comp.text, gt)
            records.append({
                "question": q,
                "generation": comp.text,
                "ground_truth": gt,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
            })
        return records


# The three mutually exclusive outcome categories.
def categorize(record: dict) -> str:
    if record["format_reward"] == 0.0:
        return "format_wrong"                      # format_wrong
    if record["answer_reward"] == 1.0:
        return "format_ok_answer_ok"               # format_ok_answer_ok
    return "format_ok_answer_wrong"                # format_ok_answer_wrong


CATEGORY_LABELS = {
    "format_ok_answer_ok": "format ok & answer ok",
    "format_ok_answer_wrong": "format ok & answer wrong",
    "format_wrong": "format wrong (no </think> <answer>...</answer>)",
}


def analyze(prompt_type: str, records: list[dict], n_examples: int = 10, out_path: str = None) -> None:
    n = len(records)
    buckets = {"format_ok_answer_ok": [], "format_ok_answer_wrong": [], "format_wrong": []}
    for r in records:
        buckets[categorize(r)].append(r)

    answer_rate = sum(r["answer_reward"] for r in records) / n
    format_rate = sum(r["format_reward"] for r in records) / n

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"[{prompt_type}]  answer accuracy: {answer_rate:.4f}   format rate: {format_rate:.4f}   (n={n})")
    lines.append("-" * 80)
    for cat in ("format_ok_answer_ok", "format_ok_answer_wrong", "format_wrong"):
        items = buckets[cat]
        lines.append(f"  {CATEGORY_LABELS[cat]:<40} {len(items):>5}  ({len(items)/n:.2%})")

    # Show a few examples from each category.
    for cat in ("format_ok_answer_ok", "format_ok_answer_wrong", "format_wrong"):
        items = buckets[cat]
        if not items:
            continue
        lines.append(f"\n  ---- examples: {CATEGORY_LABELS[cat]} ----")
        for r in items[:n_examples]:
            gen = r["generation"].strip().replace("\n", " ")
            if len(gen) > 400:
                gen = gen[:400] + " ...[truncated]"
            lines.append(f"    GT={r['ground_truth']!r}")
            lines.append(f"    generation: {gen}")
            lines.append("")

    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


def main():
    # Read data.
    data_path = "data/gsm8k/test.jsonl"
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Extract ground-truth answers (text after the final "#### ").
    for item in data:
        item["ground_truth"] = item["answer"].split("#### ")[-1].strip()

    questions = [item["question"] for item in data]
    ground_truths = [item["ground_truth"] for item in data]

    os.makedirs("eval_results", exist_ok=True)
    for prompt_type in ("question_only", "r1_zero", "r1_zero_three_shot"):
        grader = prompt_grader(prompt_type)
        records = grader.grade_all(questions, ground_truths)

        # Dump full per-example records for later inspection / the write-up.
        out_path = os.path.join("eval_results", f"{prompt_type}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps({**r, "category": categorize(r)}, ensure_ascii=False) + "\n")

        analysis_path = os.path.join("eval_results", f"{prompt_type}_analysis.txt")
        analyze(prompt_type, records, out_path=analysis_path)


if __name__ == "__main__":
    main()
