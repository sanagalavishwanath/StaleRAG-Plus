from __future__ import annotations

import math

from stalerag.models.base import ReaderOutput
from stalerag.schema import Example, Passage

INSTRUCTION = (
    "Answer the question using the passages when they are provided. "
    "Reply with only the answer, no explanation."
)


def format_passage(index: int, passage: Passage) -> str:
    prefix = f"[{index}] ({passage.date}) " if passage.date else f"[{index}] "
    return prefix + passage.text


def build_prompt(question: str, context: list[Passage]) -> str:
    parts = [INSTRUCTION, ""]
    for index, passage in enumerate(context, start=1):
        parts.append(format_passage(index, passage))
    if context:
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


def clean_answer(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0].strip(" \"'").rstrip(".")


class HFReader:
    def __init__(
        self,
        name: str,
        model_id: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 24,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
        )
        self.model.eval()

    def answer(self, example: Example, context: list[Passage]) -> ReaderOutput:
        torch = self._torch
        messages = [{"role": "user", "content": build_prompt(example.question, context)}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            transition = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
        generated = output.sequences[0, input_ids.shape[1] :]
        special = set(self.tokenizer.all_special_ids)
        logprobs = [
            float(score)
            for token, score in zip(generated.tolist(), transition[0].tolist(), strict=True)
            if token not in special
        ]
        text = clean_answer(self.tokenizer.decode(generated, skip_special_tokens=True))
        confidence = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0
        return ReaderOutput(text, confidence)
