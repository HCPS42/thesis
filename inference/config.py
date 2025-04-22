params = {
    #"model_path": "HCPS42/simpo-v1-14b-sft",
    "model_path": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    "devices": "0",
    "stages": {
        "slow": [
            {
                "max_tokens": 24576,
                "num_tries": 8,
                "temp": 1.0,
                "top_p": 0.9,
                "min_p": 0.05,
            }
        ]
    }
}
