from prompts import create_starting_prompts
from utils import batch_extract_answers, select_answer

import os
from vllm import SamplingParams
import json


def batch_inference(prompts, model, tokenizer, params) -> tuple[list[list[dict]], list[int]]:
    lengths = [sum(len(tokenizer.encode(message["content"])) for message in prompt) for prompt in prompts]
    max_tokens = params["max_tokens"] - max(lengths)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=params["temp"],
        top_p=params["top_p"],
        min_p=params["min_p"],
        skip_special_tokens=True,
    )
    
    processed_prompts = [
        tokenizer.apply_chat_template(
            conversation=prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]

    responses = model.generate(
        prompts=processed_prompts,
        sampling_params=sampling_params,
    )

    sorted_responses = []

    for prompt, response, length in zip(prompts, responses, lengths):
        prompt.append({'role': 'assistant', 'content': response.outputs[0].text})
        length += len(response.outputs[0].token_ids)
        sorted_responses.append((length, prompt))

    sorted_responses.sort(key=lambda response: response[0])

    responses = [response for _, response in sorted_responses]

    return responses


def solve(problem, model, tokenizer, params, run_name=None, problem_id=None, debug=False):
    prompts = create_starting_prompts(problem, params["num_tries"])

    responses = batch_inference(prompts, model, tokenizer, params)

    if not debug:
        output_dir = f"logs/{run_name}/traces/{problem_id}"
        os.makedirs(output_dir, exist_ok=True)
        for i, response in enumerate(responses):
            with open(f"{output_dir}/{i}.json", "w", encoding="utf-8") as f:
                json.dump(response, f, indent=4)

    extracted_answers = batch_extract_answers(responses)
    answer = select_answer(extracted_answers)

    lengths = [sum(len(tokenizer.encode(message["content"])) for message in response) for response in responses]

    if debug:
        return responses, lengths, extracted_answers, answer

    return extracted_answers, answer, lengths