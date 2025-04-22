import logging

logger = logging.getLogger(__name__)


def create_starting_prompts(problem, num_tries):
    prompts = [
        [
            {"role": "system", "content": 
                "You are a helpful and harmless assistant. "
                "You are Qwen developed by Alibaba. "
                "You should think step-by-step. "
                "Return the final answer within \\boxed{}."
            },
            {"role": "user", "content":
                f"**Problem:** {problem}\n"
                "**Solution:**"
            }
        ]
        for _ in range(num_tries)
    ]

    logger.debug("Using prompt:")
    logger.debug("_______")
    logger.debug(prompts[0])
    logger.debug("_______")

    return prompts