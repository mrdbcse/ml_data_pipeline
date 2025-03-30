import logging
import os

import pandas as pd
import yaml

log = logging.getLogger(name="__name__")
log.setLevel(level=logging.DEBUG)
# Load parameters from params.yml

params = yaml.safe_load(open("params.yaml"))["preprocess"]
log.info(f"Params: {params}")


# Preprocess
def preprocess(input_path: str, output_path: str) -> None:
    data = pd.read_csv(filepath_or_buffer=input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(path_or_buf=output_path, header=None, index=False)

    log.info(f"Preprocessed data stored to {output_path}")


if __name__ == "__main__":
    preprocess(input_path=params["input"], output_path=params["output"])
