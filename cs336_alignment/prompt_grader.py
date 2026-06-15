import os
import json

import cs336_alignment.vllm_utils as vllm_utils
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# Define vLLM server parameters
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
model_id = "OLMo-2-0425-1B"

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
    Evaluate OLMo-2-0425-1B on GSM8K with zero-shot question_only,
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
        self.vllm_server = vllm_utils.VLLMServer(model_id=model_id, host=VLLM_HOST, port=VLLM_PORT)
        self.vllm_server.start()

    def _reward_fn(self):
        return question_only_reward_fn if self.prompt_type == "question_only" else r1_zero_reward_fn

    def grade(self, question: str, ground_truth: str) -> dict:
        """Grade a single question; returns the reward dict."""
        prompt = self.template.format(question=question)
        completions = self.vllm_server.generate_completions([prompt], self.sampling_params)
        return self._reward_fn()(completions[0].text, ground_truth)

    def grade_all(self, questions: list[str], ground_truths: list[str], batch_size: int = 64) -> list[dict]:
        """Grade a batch of questions; returns one reward dict per question."""
        prompts = [self.template.format(question=q) for q in questions]
        completions = self.vllm_server.generate_completions(prompts, self.sampling_params, batch_size=batch_size)
        reward_fn = self._reward_fn()
        return [reward_fn(comp.text, gt) for comp, gt in zip(completions, ground_truths)]


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

    for prompt_type in ("question_only", "r1_zero", "r1_zero_three_shot"):
        grader = prompt_grader(prompt_type)
        results = grader.grade_all(questions, ground_truths)
        n = len(results)
        accuracy = sum(r["answer_reward"] for r in results) / n
        format_rate = sum(r["format_reward"] for r in results) / n
        print(f"[{prompt_type}] answer accuracy: {accuracy:.4f}  format rate: {format_rate:.4f}  (n={n})")


if __name__ == "__main__":
    main()
