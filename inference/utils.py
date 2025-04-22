import os
import logging
import numpy as np
import re
import random
from collections import Counter


def extract_boxed_text(text) -> int | float:
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return np.nan
    for match in matches[::-1]:
        try:
            if int(match) == float(match):
                return int(match)
        except ValueError:
            pass
    return np.nan


def batch_extract_answers(responses) -> list[int | float]:
    answers = []
    for response in responses:
        answer = extract_boxed_text(response[-1]['content'])
        answers.append(answer)
    return answers


def select_answer(answers: list[int | float]) -> tuple[int | float, float]:
    counter = Counter()
    for answer in answers:
        if answer is not np.nan:
            counter[answer] += 1 + random.random() / 1_000
    if not counter:
        return np.nan, 0.0
    
    answer = max(counter, key=counter.get)
    max_count = counter[answer]
    return answer, max_count


def setup_logging(log_name: str = "bench.log"):
    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_name, mode='a'), logging.StreamHandler()],
    )
    