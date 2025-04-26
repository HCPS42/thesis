import os
from vllm import LLM


def get_model(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["devices"]

    model = LLM(
        model=params["model_path"],
        max_num_seqs=params["num_tries"],
        max_model_len=params["max_tokens"],
        max_seq_len_to_capture=params["max_tokens"],
        trust_remote_code=True,
        tensor_parallel_size=len(params["devices"].split(",")),
        gpu_memory_utilization=params["gpu_memory_utilization"],
        enable_prefix_caching=True,
        seed=params["seed"],
    )

    return model