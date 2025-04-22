from datasets import get_dataset
from solver import solve
from config import params

import logging
import json
from time import time
import numpy as np
import pandas as pd
import os


logger = logging.getLogger(__name__)

def get_pass(summary, answers):
    return (summary == answers[:, None]).mean()

def get_cons(final_preds, answers) -> tuple[float, tuple[int, int]]:
    sum = (final_preds == answers).sum()
    count = len(final_preds)
    return sum / count, (sum, count)

def get_pretty_summary(ids, summary, answers, final_preds) -> pd.DataFrame:
    num_tries = summary.shape[1]
    df = pd.DataFrame(summary, columns=[f"try {i}" for i in range(num_tries)])
    df = pd.concat(
        [
            pd.DataFrame(answers[:, None], columns=["answer"]),
            pd.DataFrame(final_preds[:, None], columns=["consensus"]),
            df,
        ],
        axis="columns",
    )
    df["id"] = ids
    df = df.set_index("id")
    return df

def get_lengths_df(ids, lengths) -> pd.DataFrame:
    num_tries = len(lengths[0])
    df = pd.DataFrame(lengths, columns=[f"try {i}" for i in range(num_tries)])
    df["id"] = ids
    df = df.set_index("id")
    return df

def get_times_df(ids, times) -> pd.DataFrame:
    df = pd.DataFrame({"id": ids, "time": times})
    df = df.set_index("id")
    return df

def run_bench(
    model,
    dataset_name: str,
    tokenizer,
    run_name: str,
    problem_id: int = None,
):  
    logger.debug("starting ...")
    logger.debug(f"run_name: {run_name}")
    logger.debug(f"dataset_name: {dataset_name}")
    logger.debug(f"problem_id: {problem_id}")

    config_dict = {
        "params": params
    }

    if os.path.exists(f"logs/{run_name}/config.json"):
        os_config = json.load(open(f"logs/{run_name}/config.json"))
        assert sorted(os_config.keys()) == sorted(config_dict.keys())
        for k in os_config.keys():
            assert os_config[k] == config_dict[k], f"{k}: {os_config[k]} != {config_dict[k]}"

    with open(f"logs/{run_name}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    dataset = get_dataset(dataset_name)

    if problem_id is not None:
        dataset = dataset.filter(lambda x: x["id"] == problem_id)

    all_preds, all_final_preds, all_lengths, all_times = [], [], [], []
    ids = []
    for problem in dataset:
        logger.debug(f"PROBLEM: {problem['id']}")
        ids.append(problem["id"])

        if os.path.exists(f"logs/{run_name}/summary.csv"):
            summary_df = pd.read_csv(f"logs/{run_name}/summary.csv")
            summary_df.set_index("id", inplace=True)
        else:
            summary_df = None

        if summary_df is not None and problem["id"] in summary_df.index:
            logger.debug(f"Found existing results for problem {problem['id']}. Skipping...")

            lengths_df = pd.read_csv(f"logs/{run_name}/lengths.csv")    
            lengths_df.set_index("id", inplace=True)

            times_df = pd.read_csv(f"logs/{run_name}/times.csv")
            times_df.set_index("id", inplace=True)

            id = problem["id"]
            all_preds.append(list(
                map(
                    lambda x: x[1],
                    filter(
                        lambda x: x[0].startswith("try"),
                        summary_df.loc[id].to_dict().items()
                    )
                )
            ))
            print(np.array(all_preds).shape)
            all_final_preds.append(summary_df.loc[id, "consensus"].item())
            all_lengths.append(lengths_df.loc[id].tolist())
            all_times.append(times_df.loc[id, "time"].item())
            continue

        start_time = time()
        preds, final_pred, lengths = solve(
            problem=problem["problem"],
            model=model,
            tokenizer=tokenizer,
            params=params,
            speed="slow",
            run_name=run_name,
            problem_id=problem["id"]
        )
        end_time = time()
        time_elapsed = int(end_time - start_time)
        logger.debug(f"Time elapsed: {time_elapsed // 60} min {time_elapsed % 60} sec")

        logger.debug(f"answer: {problem['answer']}")
        logger.debug(f"consensus: {final_pred}")
        logger.debug(f"predictions: {preds}")
        logger.debug(f"lengths (tokens): {lengths}")

        all_preds.append(preds)
        all_final_preds.append(final_pred)
        all_lengths.append(lengths)
        all_times.append(time_elapsed)

        summary = np.array(all_preds)
        final_preds = np.array(all_final_preds)
        answers = np.array([task["answer"] for task in dataset], dtype=np.int64)[:len(all_preds)]
 
        logger.debug(f"Accuracy (pass@1): {get_pass(summary, answers):.4f}")
        cons, (sum, count) = get_cons(final_preds, answers)
        logger.debug(f"Majority vote (cons@{len(preds)}): {cons:.4f} ({sum}/{count})")

        summary_df = get_pretty_summary(ids, summary, answers, final_preds)
        summary_df.to_csv(f"logs/{run_name}/summary.csv")

        lengths_df = get_lengths_df(ids, all_lengths)
        lengths_df.to_csv(f"logs/{run_name}/lengths.csv")

        times_df = get_times_df(ids, all_times)
        times_df.to_csv(f"logs/{run_name}/times.csv")

    logger.debug("Finished.")