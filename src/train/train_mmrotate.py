import os
from pathlib import Path

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "rotated_retinanet"
    / "rotated_retinanet_iopa.py"
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("RTMDet-tiny-R — IOPA TRAINING")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nConfig:")
    print(CONFIG_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found:\n{CONFIG_PATH}"
        )

    # Register MMDetection modules
    register_all_modules_mmdet(
        init_default_scope=False
    )

    # Register MMRotate modules
    register_all_modules(
        init_default_scope=False
    )

    # Load configuration
    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    # Make sure MMRotate is the default scope
    cfg.default_scope = 'mmrotate'

    print("\nConfiguration loaded successfully.")

    print("\nTraining configuration:")
    print(f"  Epochs       : {cfg.train_cfg.max_epochs}")
    print(f"  Batch size   : {cfg.train_dataloader.batch_size}")
    print(f"  Workers      : {cfg.train_dataloader.num_workers}")
    print(f"  Image size   : 640 x 640")
    print(f"  Classes      : {len(cfg.classes)}")
    print(f"  Learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"  Device       : CUDA")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    runner = Runner.from_cfg(cfg)

    runner.train()


if __name__ == '__main__':
    main()