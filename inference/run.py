from config import params

import logging
import sys
import argparse
import os
import time
import pandas as pd


os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
os.environ["VLLM_FLASHINFER_FORCE_TENSOR_CORES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DO_NOT_TRACK"] = "1"

os.environ["HF_HOME"] = "~/.cache/huggingface"
#os.environ["HF_HOME"] = "/workspace/.hf_cache"

pd.set_option('display.max_colwidth', None)

start_time = time.time()


from model import get_model
from eval import run_bench
from utils import setup_logging
from our_datasets import dataset_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", choices=dataset_map.keys(), default="eval")
    parser.add_argument("--problem-id", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name
    assert run_name is not None, "run_name is required"

    if os.path.exists(f"logs/{run_name}") and not args.append:
        raise FileExistsError(f"The directory logs/{run_name} already exists. Change the run_name.")

    setup_logging(f"logs/{run_name}/terminal.log")
    logger = logging.getLogger(__name__)

    command = " ".join(sys.argv)
    logger.debug(f"Command: {command}")

    pid = os.getpid()
    logger.debug(f"Process ID: {pid}")

    dataset = args.dataset
    model = get_model(params)
    tokenizer = model.get_tokenizer()

    run_bench(
        model,
        dataset,
        tokenizer,
        run_name,
        args.problem_id,
    )

    end_time = time.time()
    spent_time = end_time - start_time

    logger.info(f"Spent time: {spent_time / 60} minutes")