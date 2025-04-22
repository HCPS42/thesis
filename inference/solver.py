from prompts import create_starting_prompts
from utils import extract_boxed_text, batch_extract_answers, select_answer

import os
from vllm import SamplingParams
import numpy as np
import json


def batch_inference(prompts, model, tokenizer, stage) -> tuple[list[list[dict]], list[int]]:
    
    lengths = [sum(len(tokenizer.encode(message["content"])) for message in prompt) for prompt in prompts]
    max_tokens = stage["max_tokens"] - max(lengths)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=stage["temp"],
        top_p=stage["top_p"],
        min_p=stage["min_p"],
        skip_special_tokens=True,
    )

    first_stage = True
    for message in prompts[0]:
        if message["role"] == "assistant":
            first_stage = False
            break
    
    processed_prompts = [
        tokenizer.apply_chat_template(
            conversation=prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]

    if not first_stage:
        for i in range(len(processed_prompts)):
            processed_prompts[i] = processed_prompts[i][:-32]

    responses = model.generate(
        prompts=processed_prompts,
        sampling_params=sampling_params,
    )

    sorted_responses = []

    for prompt, response, length in zip(prompts, responses, lengths):
        if first_stage:
            prompt.append({'role': 'assistant', 'content': response.outputs[0].text})
        else:
            prompt[-1]['content'] += response.outputs[0].text

        length += len(response.outputs[0].token_ids)

        sorted_responses.append((length, prompt))

    sorted_responses.sort(key=lambda response: response[0])

    responses = [response for _, response in sorted_responses]

    return responses


def solve(problem, model, tokenizer, params, speed, run_name=None, problem_id=None, debug=False):
    prompts = create_starting_prompts(problem, params["stages"][speed][0]["num_tries"])

    finished_responses = []

    for stage in params["stages"][speed]:
        if len(prompts) > stage["num_tries"]:
            prompts = prompts[-stage["num_tries"]:]

        responses = batch_inference(prompts, model, tokenizer, stage)

        prompts = []

        for response in responses:
            if not np.isnan(extract_boxed_text(response[-1]["content"])):
                finished_responses.append(response)
            else:
                prompts.append(response)

        if not prompts:
            break

        extracted_answers = batch_extract_answers(finished_responses)
        answer, cnt = select_answer(extracted_answers)

    if not debug:
        output_dir = f"logs/{run_name}/traces/{problem_id}"
        os.makedirs(output_dir, exist_ok=True)
        for i, response in enumerate(finished_responses):
            with open(f"{output_dir}/{i}.json", "w", encoding="utf-8") as f:
                json.dump(response, f, indent=4)
    
    extracted_answers = batch_extract_answers(finished_responses)
    answer, cnt = select_answer(extracted_answers)

    lengths = [sum(len(tokenizer.encode(message["content"])) for message in response) for response in finished_responses]

    if debug:
        return finished_responses, lengths, extracted_answers, answer

    max_num_tries = 0
    for s in params["stages"]:
        for stage in params["stages"][s]:
            max_num_tries = max(max_num_tries, stage["num_tries"])  

    while len(extracted_answers) < max_num_tries:
        extracted_answers.append(np.nan)
        lengths.append(-1)

    return extracted_answers, answer, lengths