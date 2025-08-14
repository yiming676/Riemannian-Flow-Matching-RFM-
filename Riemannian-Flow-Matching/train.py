"""

python train-cifar10.py experiment=embed128_flattours \
       ++embed128_csv_path=/root/autodl-tmp/mapper/train_embeds128.csv
"""

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import os.path as osp
import sys
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import json
from glob import glob
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.plugins.environments import SLURMEnvironment

from manifm.model_pl import Embed128DataModule
from manifm.model_pl import ManifoldFMLitModule

torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    if cfg.get("seed", None) is not None:
        # pl.utilities.seed.seed_everything(cfg.seed)
        from lightning_fabric.utilities.seed import seed_everything
        seed_everything(cfg.seed)

    print(OmegaConf.to_yaml(cfg))
    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"{props.name} \t Memory: {props.total_memory / (1024 ** 3):.2f}GB")

    keys = ["SLURM_NODELIST", "SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_JOB_NAME", "SLURM_PROCID", "SLURM_LOCALID",
            "SLURM_NODEID"]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))
    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)
    log.info(f"CWD: {os.getcwd()}")

    datamodule = Embed128DataModule(cfg)

    model = ManifoldFMLitModule(cfg)
    print(model)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            monitor="val/loss_best",
            mode="min",
            filename="epoch-{epoch:03d}_step-{global_step}_loss-{val/loss_best:.6f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=cfg.get("ckpt_every", None),
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(monitor="val/loss_best", patience=cfg.early_stopping_patience, mode="min"),
    ]

    # slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)
    slurm_plugin = SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [
        CSVLogger(save_dir=".", name="csv_logs"),
        TensorBoardLogger(save_dir=".", name="tb_logs"),
    ]
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".", name=f"{cfg.data}_{now}", project="ManiFM",
                log_model=False, config=cfg_dict, resume="allow"
            )
        )

    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        logger=loggers,
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        num_sanity_val_steps=0,
    )

    ckpt_path = cfg.get("resume", None)
    if ckpt_path is None:
        last_ckpt = glob(osp.join("checkpoints", "last.ckpt"))
        if len(last_ckpt) > 0:
            ckpt_path = last_ckpt[0]
            log.info(f"Found last.ckpt, resuming training from: {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Starting testing!")

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info(f"Best ckpt path: {ckpt_path}")

    metric_dict = {k: float(v) for k, v in trainer.callback_metrics.items()}
    with open("metrics.json", "w") as fout:
        json.dump(metric_dict, fout)

    return metric_dict


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        log.error(traceback.format_exc())
        sys.exit(1)