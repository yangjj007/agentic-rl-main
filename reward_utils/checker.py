import re
from typing import Optional

from client_utils.openai_api import OpenAIClient
from data_utils.aokvqa.evaluator import eval_aokvqa_direct
from data_utils.chart.evaluator import eval_one_chart
from data_utils.commom_util import prompt_ic

from reward_utils.template_pool import (
    TemplatePool,
    compare_templates_via_client,
    update_best_template_if_different as _pool_update_best_template,
)

_DEFAULT_POOL = TemplatePool()


def update_best_template_if_different(client, system_prompt, new_template: str):
    """Backward-compatible wrapper around shared TemplatePool."""
    _pool_update_best_template(
        client,
        system_prompt,
        new_template,
        pool=_DEFAULT_POOL,
    )


class RewardCalculator:
    """
    A class to calculate various rewards for a model's response.
    Encapsulates logic for answer correctness, format adherence, and thinking quality.
    """

    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        """
        Initializes the RewardCalculator.

        Args:
            answer_flag (str): The keyword that separates reasoning from the final answer.
        """
        answer_flag = RL_CONFIG["answer_flag"]
        self.answer_flag = answer_flag.lower()
        self.count_pattern = re.compile(f'(?i){re.escape(self.answer_flag)}')
        if CLIENT_CONFIG['client_type'] == 'openai':
            if CLIENT_CONFIG['init_port'] is not None:
                num_server = int(CLIENT_CONFIG['num_server'])
                server_id = gpu_id % num_server
                CLIENT_CONFIG['api_base'] = CLIENT_CONFIG['api_base'] % str(CLIENT_CONFIG['init_port'] + server_id)
            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def get_answer_reward(self, response: str, reference_answer: str, task: str, gpu_id=None, answer_type=None) -> float:
        """
        Calculates the correctness reward for the answer.
        Returns 1.0 if correct, 0.0 otherwise.
        """
        try:
            if 'chart' in task:
                # Assuming eval_one_chart returns a float (e.g., 1.0 for correct, 0.0 for incorrect)
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'math_lm' in task:
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'world' in task:
                reward = eval_aokvqa_direct(response, reference_answer)
                return float(reward)

            else:
                raise ValueError(f"Task '{task}' not supported for answer reward.")
        except Exception as e:
            # Catch specific exceptions and log them for better debugging.
            print(f"An error occurred during answer reward calculation: {e}")
            return 0.0

    def get_format_reward(self, response: str, min_thinking_length: int | None = None, task: str = "") -> float:
        """
        Calculates the format reward based on two criteria:
        1. The 'answer:' flag must appear exactly once.
        2. The preceding 'thinking' text must meet a minimum length.

        Returns 1.0 if the format is correct, 0.0 otherwise.
        """
        min_len = 0 if min_thinking_length is None else min_thinking_length
        if "chart" in (task or ""):
            from reward_utils.format_checks import evaluate_format_reward

            return evaluate_format_reward(
                response,
                self.answer_flag,
                self.count_pattern,
                min_thinking_length=min_len,
                task=task,
            )

        # 1. Check if the answer flag appears exactly once.
        if len(self.count_pattern.findall(response)) != 1:
            return 0.0

        # 2. Check if the 'thinking' part has sufficient length.
        thinking = response.lower().split(self.answer_flag)[0]
        if len(thinking.strip()) < min_len:
            return 0.0

        return 1.0

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        """
        Generates a prompt for an LLM to evaluate the quality of the 'thinking' process.

        This function prepares the input; an external LLM call would be needed to get a score.

        Returns:
            A formatted prompt string, or None if the task is unsupported.
        """

        def get_score(level_string):
            if "low" in level_string:
                return 0
            elif "medium" in level_string:
                return 0.5
            elif "high" in level_string:
                return 1
            else:
                # Handle unknown input
                return 0  # or return -1, or raise an error

        system_prompt = None
        if 'medical' in task:
            system_prompt = 'You are a seasoned professional in the field of medical image analysis, demonstrating exceptional expertise and insight into complex medical imaging data. Your output should be only judgement, without any additional text or explanation.'
        elif 'math' in task:
            system_prompt = 'You are a seasoned professional in the field of mathematics, demonstrating exceptional expertise and insight into complex mathematical problems. Your output should be only judgement, without any additional text or explanation.'
        elif 'chart' in task:
            system_prompt = 'You are a seasoned professional in the field of chart analysis, demonstrating exceptional expertise and insight into complex chart data. Your output should be only judgement, without any additional text or explanation.'
        elif 'world' in task:
            system_prompt = 'You are a seasoned professional in the field of world knowledge and image analysis, demonstrating exceptional expertise and insight into complex real-world scenarios. Your output should be only judgement, without any additional text or explanation.'
        else:
            Exception('Unknown expert task')

        try:
            thinking = response.lower().split(self.answer_flag)[0].strip()
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt, max_tokens=5000)

            if 'chart' in task or 'world' in task:
                if 'chart' in task:
                    from data_utils.chart.prompts import prompt_thinking_reward, prompt_template
                else:
                    from data_utils.aokvqa.prompts import prompt_thinking_reward, prompt_template
                # Construct the final prompt for the evaluator model.
                evaluation_prompt = prompt_thinking_reward % (in_context_example, question, answer, thinking)
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=10)
                reward = get_score(output)

                if reward == 1:
                    template_prompt = prompt_template % thinking
                    ext_template = self.client.get_completion(template_prompt, system_prompt=system_prompt, max_tokens=512)
                    if "none" not in ext_template.strip().lower():
                        _DEFAULT_POOL.maybe_update(
                            ext_template,
                            lambda cur, new: compare_templates_via_client(
                                self.client, system_prompt, cur, new
                            ),
                        )
                return reward
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"An error occurred during thinking reward prompt generation: {e}")
            return None




class RewardCalculatorLocal:
    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        # ... other initialization code ...
        self.answer_flag = RL_CONFIG["answer_flag"].lower()
        self.count_pattern = re.compile(f'(?i){re.escape(self.answer_flag)}')

        # Load spaCy's small English model (file-locked; safe under multi-GPU launch)
        from reward_utils.spacy_model import load_spacy_english

        self.nlp = load_spacy_english()

        # Define the POS tags we consider "important":
        # NOUN, PROPN, VERB, ADJ, NUM
        # You can adjust this list as needed
        self.important_pos_tags = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}

    def _preprocess_text_pos(self, text: str) -> set[str]:
        """
        Use part-of-speech tagging to extract keywords.
        """
        doc = self.nlp(text.lower())
        keywords = set()
        for token in doc:
            # Keep only tokens with important POS tags, and ensure they are not stopwords or punctuation
            if token.pos_ in self.important_pos_tags and not token.is_stop and not token.is_punct:
                # Use lemma_ to get the base form (e.g., 'sales' -> 'sale')
                keywords.add(token.lemma_)
        return keywords

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        try:
            thinking_part = response.lower().split(self.answer_flag)[0].strip()
            if not thinking_part:
                return 0.0

            # Use the new POS-based method
            thinking_tokens = self._preprocess_text_pos(thinking_part)
            reference_tokens = self._preprocess_text_pos(hint)

            # Compute intersection
            common_tokens = thinking_tokens.intersection(reference_tokens)

            # Precision:
            # Of all tokens generated by the model, how many are correct (appear in hint)?
            precision = len(common_tokens) / (len(thinking_tokens) + 1e-6)

            # Recall:
            # Of all correct tokens in the hint, how many were found by the model?
            recall = len(common_tokens) / (len(reference_tokens) + 1e-6)

            # F1-score
            if precision + recall == 0:
                return 0.0

            f1_score = 2 * (precision * recall) / (precision + recall)

            return f1_score

        except Exception as e:
            print(f"An error occurred during local thinking reward calculation: {e}")
            return 0.0

    def get_answer_reward(self, response: str, reference_answer: str, task: str, gpu_id=None, answer_type=None) -> float:
        """
        Calculates the correctness reward for the answer.
        Returns 1.0 if correct, 0.0 otherwise.
        """
        try:
            if 'chart' in task:
                # Assuming eval_one_chart returns a float (e.g., 1.0 for correct, 0.0 for incorrect)
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'math_lm' in task:
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'world' in task:
                reward = eval_aokvqa_direct(response, reference_answer)
                return float(reward)
            else:
                raise ValueError(f"Task '{task}' not supported for answer reward.")
        except Exception as e:
            # Catch specific exceptions and log them for better debugging.
            print(f"An error occurred during answer reward calculation: {e}")
            return 0.0

    def get_format_reward(self, response: str, min_thinking_length: int | None = None, task: str = "") -> float:
        """
        Calculates the format reward based on two criteria:
        1. The 'answer:' flag must appear exactly once.
        2. The preceding 'thinking' text must meet a minimum length.

        Returns 1.0 if the format is correct, 0.0 otherwise.
        """
        min_len = 0 if min_thinking_length is None else min_thinking_length
        if "chart" in (task or ""):
            from reward_utils.format_checks import evaluate_format_reward

            return evaluate_format_reward(
                response,
                self.answer_flag,
                self.count_pattern,
                min_thinking_length=min_len,
                task=task,
            )

        # 1. Check if the answer flag appears exactly once.
        if len(self.count_pattern.findall(response)) != 1:
            return 0.0

        # 2. Check if the 'thinking' part has sufficient length.
        thinking = response.lower().split(self.answer_flag)[0]
        if len(thinking.strip()) < min_len:
            return 0.0

        return 1.0
