"""Run a real same-GPU visual checker/refiner audit on a ChartQA image.

This is deliberately a diagnostic, not a training entry point.  It loads the
same local student/teacher pair used by the OPD smoke recipe, feeds a real
ChartQA image through the image-primary checker and the no-gold refiner, and
writes the normal ``visual_supervision`` JSONL artifacts.

Example:
  /data/junjie/.miniforge3/envs/eval3d/bin/python \
    scripts/audit_visual_teacher_eval3d.py \
    --output-dir outputs/teacher-visual-eval3d-audit-latest
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config
from main import load_model_and_processor, load_teacher_model
from reward_utils.compute_rewards import calculate_rewards_sequential, refine_context_sequential
from reward_utils.visual_supervision_factory import build_visual_supervision


def _chartqa_answer(label) -> str:
    if isinstance(label, list):
        return str(label[0]) if label else ""
    text = str(label or "").strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    return str(parsed[0]) if isinstance(parsed, list) and parsed else str(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/teacher-visual-eval3d-audit-latest",
        help="Directory for normal visual_supervision JSONL artifacts.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config("opd_7b_dyme_probe_image_checker")
    model_cfg = dict(cfg["model"])
    model_cfg.update(
        pretrained_model_path="/data/junjie/models/opd_eval3d/student-0.5b",
        teacher_model_path="/data/junjie/models/opd_eval3d/teacher-7b",
        teacher_device_map="cuda:0",
        use_flash_attention_2=False,
    )
    visual_cfg = cfg["opsd"]["visual_supervision"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[VISUAL-AUDIT] loading local student and same-card frozen teacher", flush=True)
    student, processor = load_model_and_processor(model_cfg)
    teacher = load_teacher_model(model_cfg, local_rank=0, num_gpus=1)
    # Match the real OPD recipe's colocated two-model placement.  The audit
    # does not backpropagate through the student, but keeping it resident on
    # cuda:0 verifies the checker/refiner contract under actual card pressure.
    student.to("cuda:0")
    student.eval()
    print(
        "[VISUAL-AUDIT] placement "
        f"student={next(student.parameters()).device} "
        f"teacher={next(teacher.parameters()).device}",
        flush=True,
    )

    stream = load_dataset("HuggingFaceM4/ChartQA", split="train", streaming=True)
    row = next(item for idx, item in enumerate(stream) if idx == args.sample_index)
    question = str(row["query"])
    answer = _chartqa_answer(row["label"])
    image = row["image"]
    sample = {
        "image": image,
        "question_wo_prompt": question,
        "prompt": question,
        "hint": "Use only visible chart evidence and explain the comparison.",
    }
    good = (
        "Goal: Answer the chart question.\n"
        "Observation: The required value and year are visible on the plotted series.\n"
        "Reasoning: I compare the requested year with the labelled favorable value in the chart.\n"
        "Conclusion: The visible plotted value supports the stated response.\n"
        f"Answer: {answer}"
    )
    bad = (
        "Goal: Guess without reading the chart.\n"
        "Observation: I have no visual evidence.\n"
        "Reasoning: I choose an unrelated answer.\n"
        "Conclusion: This is unsupported.\n"
        "Answer: Maybe"
    )

    checker, refiner, meta = build_visual_supervision(
        cfg["rl"], cfg["client"], cfg["opsd"], gpu_id=0,
        teacher_model=teacher, processor=processor,
    )
    print(
        "[VISUAL-AUDIT] components "
        f"checker={type(checker).__name__} refiner={type(refiner).__name__} "
        f"teacher_bound={getattr(checker, '_teacher_model', None) is teacher}",
        flush=True,
    )
    samples, images, questions = [sample, sample], [image, image], [question, question]
    checker.begin_generate_batch(
        samples=samples, images=images, questions=questions,
        global_step=1, output_dir=str(output_dir),
    )
    refiner.begin_generate_batch(
        samples=samples,
        images=images,
        questions=questions,
        global_step=1,
        output_dir=str(output_dir),
        recorder=getattr(checker, "_recorder", None),
        ic_cache=getattr(checker, "_ic_cache", None),
        skip_cold_start=False,
    )
    try:
        all_rewards, format_rewards, answer_rewards, thinking_rewards = calculate_rewards_sequential(
            checker,
            {
                "response": [good, bad], "prompt": [question, question],
                "answer": [answer, answer], "hints": [sample["hint"], sample["hint"]],
            },
            gpu_id=0,
            task="chart",
        )
        refined = refine_context_sequential(
            refiner, [question], [sample["hint"]], [answer], "chart", 0,
        )
    finally:
        stats = checker.end_generate_batch()
        refiner.end_generate_batch()

    payload = {
        "question": question,
        "reference_answer": answer,
        "good_response": good,
        "bad_response": bad,
        "rewards": {
            "all": all_rewards,
            "format": format_rewards,
            "answer": answer_rewards,
            "thinking": thinking_rewards,
        },
        "refined": refined,
        "reference_in_refined": answer.lower() in str(refined[0]).lower(),
        "visual_stats": stats,
        "artifacts": str(output_dir / "visual_supervision" / "step_1" / "rank0.jsonl"),
    }
    print("[VISUAL-AUDIT] result " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
