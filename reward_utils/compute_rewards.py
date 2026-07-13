import concurrent.futures
from typing import List, Dict, Any, Optional
from .checker import RewardCalculator

def split_initial_context(text: str):
    text = text.lower()
    flag = 'answer:'
    if flag in text:
        ans = text.split(flag)[-1].strip()
        context = text.split(flag)[0].strip()
        ans = ans.strip('.')
    else:
        context = text
        ans = text
    return context, ans

def calculate_rewards_in_parallel(
    checker: RewardCalculator,
    batch_data: Dict[str, Any],
    gpu_id: int,
    num_threads: int = 8,
    task='chart'):
    """
    Calculates accuracy rewards for a batch of data in parallel using a thread pool.

    Args:
        batch_data: A dictionary containing lists of data, including 'response',
                    'prompt', 'image', 'answer', and an optional 'tp' (answer_type).
        gpu_id: The ID of the GPU to be used for processing.
        num_threads: The number of parallel threads to use.

    Returns:
        A list of calculated reward scores for each item in the batch.
    """
    # Extract lists of data from the input dictionary
    responses = batch_data['response']
    predictions = []
    for r in responses:
        c, p = split_initial_context(r)
        predictions.append(p)
    prompts = batch_data['prompt']
    # questions = batch_data['question']
    answers = batch_data['answer']
    hints = batch_data['hints'] if 'hints' in batch_data else [""] * len(responses)
    num_samples = len(responses)

    # Safely get 'answer_types', providing a list of Nones as a default
    # This fixes a bug in the original code.
    answer_types = batch_data.get('tp', [None] * num_samples)

    # Prepare the arguments for each task by zipping the data together.
    # This creates an iterator of tuples, where each tuple contains all args for one call.
    in_answers = answers
    if 'world' in task:
        in_answers = batch_data['direct_answers']
    task_answer_args = zip(
        predictions,
        in_answers,
        [task] * num_samples,
        # [gpu_id] * num_samples,
        # answer_types,
        # hints
    )
    task_thinking_args = zip(
        responses,
        prompts,
        answers,
        hints,
        [task] * num_samples
    )

    # Use a ThreadPoolExecutor to process the data in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Instead of a separate function, use a lambda to unpack the arguments.
        # The '*' operator unpacks each tuple from task_args into positional arguments
        # for the get_acc_reward function.

        format_rewards = list(executor.map(lambda r: checker.get_format_reward(r, task=task), responses))
        answer_rewards = list(executor.map(lambda args: checker.get_answer_reward(*args), task_answer_args))
        thinking_rewards = list(executor.map(
            lambda args: checker.get_thinking_reward_prompt(*args), task_thinking_args))

        rewards = [0 if f == 0 else f + a + t for f, a, t in zip(format_rewards, answer_rewards, thinking_rewards)]

    return rewards, format_rewards, answer_rewards, thinking_rewards


def calculate_rewards_sequential(
    checker,
    batch_data: Dict[str, Any],
    gpu_id: int,
    task='chart',
):
    """Sequential reward path for GPU-backed teacher checker (not thread-safe)."""
    responses = batch_data['response']
    predictions = []
    for r in responses:
        c, p = split_initial_context(r)
        predictions.append(p)
    prompts = batch_data['prompt']
    answers = batch_data['answer']
    hints = batch_data['hints'] if 'hints' in batch_data else [""] * len(responses)
    num_samples = len(responses)
    in_answers = answers
    if 'world' in task:
        in_answers = batch_data['direct_answers']

    format_rewards = []
    answer_rewards = []
    thinking_rewards = []

    if hasattr(checker, "prepare_thinking_jobs") and hasattr(checker, "batch_score_thinking"):
        jobs = checker.prepare_thinking_jobs(responses, prompts, answers, hints, task)
        if jobs:
            checker.batch_score_thinking(jobs, task)

    for i in range(num_samples):
        checker._current_sample_idx = i  # noqa: SLF001 — teacher checker batch context
        format_rewards.append(checker.get_format_reward(responses[i], task=task))
        answer_rewards.append(
            checker.get_answer_reward(predictions[i], in_answers[i], task)
        )
        cache = getattr(checker, "_thinking_score_cache", None)
        if cache is not None and i in cache:
            thinking_rewards.append(cache[i])
        else:
            thinking_rewards.append(
                checker.get_thinking_reward_prompt(
                    responses[i], prompts[i], answers[i], hints[i], task
                )
            )

    rewards = [
        0 if f == 0 else f + a + t
        for f, a, t in zip(format_rewards, answer_rewards, thinking_rewards)
    ]
    return rewards, format_rewards, answer_rewards, thinking_rewards


def refine_context_in_parallel(
    refiner,
    questions: List[str],
    hints: List[str],
    reference_answers: List[str],
    task,
    gpu_id: int,
    num_threads: int = 8):
    """
    Refines contexts for a batch of data in parallel using a thread pool.

    Args:
        questions: A list of questions.
        hints: A list of hints corresponding to each question.
        reference_answers: A list of reference answers.
        tasks: A list of task types corresponding to each question.
        gpu_id: The ID of the GPU to be used for processing.
        num_threads: The number of parallel threads to use.

    Returns:
        A list of refined contexts for each question.
    """
    num_samples = len(questions)
    tasks = [task] * num_samples
    # Prepare the arguments for each task by zipping the data together.
    task_args = zip(
        questions,
        hints,
        reference_answers,
        tasks,
        [gpu_id] * num_samples
    )

    # Use a ThreadPoolExecutor to process the data in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        refined_contexts = list(executor.map(
            lambda args: refiner.refine_hint(*args), task_args
        ))

    return refined_contexts


def refine_context_sequential(
    refiner,
    questions: List[str],
    hints: List[str],
    reference_answers: List[str],
    task,
    gpu_id: int,
):
    """Sequential refine path for GPU-backed teacher refiner (optional per-batch dedupe)."""
    from reward_utils.visual_batch_ops import refine_dedupe_key

    visual_cfg = getattr(refiner, "visual_config", {}) or {}
    dedupe = visual_cfg.get("dedupe_per_batch", True)
    images = getattr(refiner, "_batch_images", None) or []

    if not dedupe:
        if hasattr(refiner, "batch_refine_hints"):
            from reward_utils.visual_refiner_teacher import _RefinerJob

            jobs = [
                _RefinerJob(
                    sample_idx=i,
                    question=q,
                    hint=h,
                    reference_answer=a,
                )
                for i, (q, h, a) in enumerate(zip(questions, hints, reference_answers))
            ]
            results = refiner.batch_refine_hints(jobs, task)
            return [results.get(i, hints[i]) for i in range(len(questions))]

        refined = []
        for i, (q, h, a) in enumerate(zip(questions, hints, reference_answers)):
            refiner._current_sample_idx = i  # noqa: SLF001
            refined.append(refiner.refine_hint(q, h, a, task, gpu_id))
        return refined

    n = len(questions)
    refined: list[Optional[str]] = [None] * n
    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, (q, h) in enumerate(zip(questions, hints)):
        image = images[i] if i < len(images) else None
        key = refine_dedupe_key(q, h, image)
        groups.setdefault(key, []).append(i)

    if hasattr(refiner, "batch_refine_hints"):
        from reward_utils.visual_refiner_teacher import _RefinerJob

        leader_jobs = [
            _RefinerJob(
                sample_idx=indices[0],
                question=questions[indices[0]],
                hint=hints[indices[0]],
                reference_answer=reference_answers[indices[0]],
            )
            for indices in groups.values()
        ]
        batch_results = refiner.batch_refine_hints(leader_jobs, task)
        for indices in groups.values():
            i0 = indices[0]
            result = batch_results.get(i0, hints[i0])
            for i in indices:
                refined[i] = result
                if i != i0 and hasattr(refiner, "record_refiner_dedupe"):
                    refiner.record_refiner_dedupe(
                        sample_idx=i,
                        result=result,
                        hint=hints[i],
                        source_idx=i0,
                    )
        return refined

    for indices in groups.values():
        i0 = indices[0]
        refiner._current_sample_idx = i0  # noqa: SLF001
        result = refiner.refine_hint(
            questions[i0],
            hints[i0],
            reference_answers[i0],
            task,
            gpu_id,
        )
        for i in indices:
            refined[i] = result
            if i != i0 and hasattr(refiner, "record_refiner_dedupe"):
                refiner.record_refiner_dedupe(
                    sample_idx=i,
                    result=result,
                    hint=hints[i],
                    source_idx=i0,
                )
    return refined