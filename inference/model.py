import os
from vllm import LLM


def get_model(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["devices"]

    max_num_tries = 0
    for speed, stages in params["stages"].items():
        for stage in stages:
            max_num_tries = max(max_num_tries, stage["num_tries"])
    
    max_max_tokens = 0
    for speed, stages in params["stages"].items():
        for stage in stages:
            max_max_tokens = max(max_max_tokens, stage["max_tokens"])

    model = LLM(
        model=params["model_path"],
        max_num_seqs=max_num_tries,
        max_model_len=max_max_tokens,
        max_seq_len_to_capture=max_max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=len(params["devices"].split(",")),
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        quantization="awq_marlin",
        seed=42,
    )

    return model