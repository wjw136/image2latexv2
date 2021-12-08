
import sys
import os
sys.path.append(os.path.abspath('./'))#包所在的根目录

from argparse import Namespace
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from image_to_latex.data import Im2Latex
from image_to_latex.lit_models import LitResNetTransformer


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()
    val=datamodule.val_dataloader()
    # print(datamodule.val_dataloader)

    # for i in val:
    #     print(i)
    #     break;
    
    lit_model = LitResNetTransformer.load_from_checkpoint('/data/zzengae/jwwang/final_project/test1/epoch=7-val_loss=0.40-val_edit_distance=0.85-val_exact_match=0.58.ckpt')
    # lit_model.load_from_checkpoint('/data/zzengae/jwwang/final_project/test/epoch=0-val_loss=1.75-val_edit_distance=0.20-val_exact_match=0.02.ckpt')

    
    # print(cfg)
    trainer = Trainer(**cfg.trainer)
    # trainer = Trainer(resume_from_checkpoint='/data/zzengae/jwwang/final_project/test/epoch=0-val_loss=1.75-val_edit_distance=0.20-val_exact_match=0.02.ckpt')

    # **cfg 命名参数 *cfg 位置参数 cfg list or dict
    # print(*cfg)
    # # print(**cfg)
    # print(cfg)

    # trainer.tune(lit_model, datamodule=datamodule) # call tune to find the lr
    # trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)
    # trainer.validate(lit_model, dataloaders=val)


if __name__ == "__main__":
    main()
