import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path("/kaggle/working/input")
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    train_df = pl.scan_parquet(Path(cfg.dir.input_dir) / "train.parquet")

    # 末尾のだけ取り出し
    valid_df = train_df.tail(cfg.exp.n_rows).collect()
    print(valid_df)

    valid_df.write_parquet(output_path / "valid.parquet")


if __name__ == "__main__":
    main()
