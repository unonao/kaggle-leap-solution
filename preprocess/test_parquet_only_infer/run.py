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

    test_df = pl.read_csv(
        Path(cfg.dir.input_dir) / "leap-atmospheric-physics-ai-climsim/test.csv"
    )
    sample_submission_df = pl.read_csv(
        Path(cfg.dir.input_dir)
        / "leap-atmospheric-physics-ai-climsim/sample_submission.csv"
    )

    test_df.write_parquet(output_path / "test.parquet")
    sample_submission_df.write_parquet(output_path / "sample_submission.parquet")


if __name__ == "__main__":
    main()
