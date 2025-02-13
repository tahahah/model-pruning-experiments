import os
import sys

from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.dcaecore.trainer import DCAETrainer, DCAETrainerConfig


def main():
    cfg: DCAETrainerConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAETrainerConfig), OmegaConf.from_cli()))
    trainer = DCAETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
