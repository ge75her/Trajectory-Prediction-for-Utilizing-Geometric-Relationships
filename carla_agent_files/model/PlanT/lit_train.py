import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

from dataloader import get_dataloader
from model.PlanT.lit_module import LitHFLM


@hydra.main(config_path=f"../config", config_name="config")
def main(cfg):

    # print config
    print(OmegaConf.to_yaml(cfg))

    # setup debug mode
    overfit = 0.0
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
        cfg.expname = "debug"
        overfit = 5  # use only 5 fixed batches for debugging

    if cfg.overfit > 0:
        overfit = cfg.overfit

    # use data caching for ML-Cloud #TODO
    shared_dict = None

    # if we use mutliple GPUs and want wandb online it does need too much 
    # time on the MLCLoud and the training freezes or is too slow
    # log only local and sync afterwards with wandb sync [OPTIONS] [PATH]
    # if cfg.gpus > 1:
    #     os.environ["WANDB_MODE"] = "offline"

    # setup logging
    pl.seed_everything(cfg.seed)

    # setup lightning logger
    csvlogger = CSVLogger(cfg.model.training.ckpt_path, "CSVLogger")
    

    # resume training
    resume_path = "/home/masterthesis/checkpoints_junc_0.2/last.ckpt"
    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    checkpoint_path = "/home/masterthesis/checkpoints_junc_0.2/last.ckpt"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="/home/masterthesis/checkpoints_junc_0.2",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=2,
    )

    train_loader, val_loader = get_dataloader(cfg, shared_dict=shared_dict)

    if cfg.model.training.pretraining_path != "none":
        GPT_model = LitHFLM.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        GPT_model = LitHFLM(cfg=cfg)


    if cfg.gpus > 1:       
        replace_sampler_ddp =True
        trainer = Trainer(
            callbacks=checkpoint_callback,
            accelerator='gpu',
            devices=[1,2],
            strategy="ddp",
            replace_sampler_ddp=replace_sampler_ddp,
            logger=[csvlogger],
            log_every_n_steps=2,
            resume_from_checkpoint=resume_path,
            check_val_every_n_epoch=2,
            max_epochs=cfg.model.training.max_epochs,
            overfit_batches=overfit,
        )
    else:
        trainer = Trainer(
            callbacks=checkpoint_callback,
            accelerator="gpu",
            devices=[0],
            logger=[csvlogger],
            log_every_n_steps=2,
            resume_from_checkpoint=resume_path,
            check_val_every_n_epoch=2,
            max_epochs=cfg.model.training.max_epochs,
            overfit_batches=overfit,
        )

    trainer.fit(GPT_model, train_loader, val_loader)

 


if __name__ == "__main__":
    main()
