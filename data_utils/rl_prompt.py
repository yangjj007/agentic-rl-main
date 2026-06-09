"""Shared RL prompt template for VQA / chart / math tasks."""
from config import CONFIG


def build_rl_prompt_template(answer_flag: str | None = None) -> str:
    """Build user prompt text before chat template is applied.

    Avoids the legacy ``"Answer: .."`` quoted placeholder, which biased the model
    to emit token 340 (``)``) immediately after ``<|im_start|>assistant``.
    """
    flag = answer_flag or CONFIG["rl"]["answer_flag"]
    return (
        "Your task is to answer the question below. "
        "Think step by step before giving your final answer. "
        f"When you are ready, end your response with a line starting with {flag} "
        "followed by your answer.\n\n"
        "Question:\n\n{question}"
    )


PROMPT_TEMPLATE = build_rl_prompt_template()
