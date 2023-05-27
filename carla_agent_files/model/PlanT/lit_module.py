import logging

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy

from model.PlanT.model import HFLM,Loss

logger = logging.getLogger(__name__)


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.last_epoch = 0
        self.cfg_train = self.cfg.model.training
        self.model = HFLM(self.cfg.model.network, self.cfg)
        # checkpoint = torch.load('/home/masterthesis/plant/checkpoints/PlanT/3x/PlanT_medium/checkpoints/epoch=047.ckpt') 
        # state_dict=checkpoint['state_dict']
        # # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[6:] # remove `model.`，
        #     new_state_dict[name] = v  
        # # load params
        # self.model.load_state_dict(new_state_dict,strict=False) 
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)
        self.cri_lane=Loss(4)
        self.cri_junc=Loss(3)
        self.val_step_outputs=[]
        self.val_step_outputs_junc=[]
        # Metrics
        # if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
        #     self.metrics_forecasting_acc = nn.ModuleList(
        #         [Accuracy() for i in range(self.model.num_attributes)]
        #     )
            

    def forward(self, x, y=None, target_point=None, light_hazard=None):
        return self.model(x, y, target_point, light_hazard)

 
    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.model.training)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y, wp, tp, light,label_lane,label_junc = batch

        # training with only waypoint loss
        if self.cfg.model.pre_training.get("pretraining", "none") == "none":
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "train/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

        elif self.cfg.model.pre_training.get("pretraining", "none") == "forecast":

            if self.cfg.model.pre_training.get("multitask", False) == True:
                # multitask training
                logits, targets, pred_wp, _,x_lane,x_junc = self(x, y, tp, light)

                loss_wp = F.l1_loss(pred_wp, wp)
                loss_lane,acc_lane,_=self.cri_lane(x_lane,label_lane)
                loss_junc,acc_junc,_=self.cri_junc(x_junc,label_junc)
                
                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_forecast = torch.mean(torch.stack(losses_forecast))

                loss_all = (
                    1                                                           * loss_wp
                    + self.cfg.model.pre_training.get("forecastLoss_weight", 0) * loss_forecast
                    +0.2*loss_lane
                    +0.2*loss_junc
                )
                self.log(
                    "train/loss_forecast",
                    loss_forecast,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/loss_wp",
                    loss_wp,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/loss_lane",
                    loss_lane,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/acc_lane",
                    acc_lane,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/acc_junc",
                    acc_junc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

            else:
                # 2 stage training (pre-training only on forecasting - no waypoint loss)
                logits, targets = self(x, y)

                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_all = torch.mean(torch.stack(losses_forecast))

            self.log(
                "train/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            # for i, name in enumerate(
            #     ["x", "y", "yaw", "speed", "extent_x", "extent_y"]
            # ):
            #     if i > self.model.num_attributes:
            #         break
            #     self.log(
            #         f"train/loss_{name}",
            #         losses_forecast[i],
            #         on_step=False,
            #         on_epoch=True,
            #         prog_bar=False,
            #         sync_dist=self.cfg.gpus > 1,
            #         batch_size=self.cfg.model.training.batch_size,
            #     )

            #     mask = targets[i].squeeze() != -999
            #     self.metrics_forecasting_acc[i](
            #         logits[i][mask], targets[i][mask].squeeze()
            #     )
            #     self.log(
            #         f"train/acc_{name}",
            #         self.metrics_forecasting_acc[i],
            #         on_step=False,
            #         on_epoch=True,
            #         prog_bar=False,
            #         sync_dist=self.cfg.gpus > 1,
            #         batch_size=self.cfg.model.training.batch_size,
            #     )

        return loss_all


    def validation_step(self, batch, batch_idx):

        if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
            x, y, wp, tp, light,label_lane,label_junc = batch

            if self.cfg.model.pre_training.get("multitask", False) == True:
                # multitask training
                _, _, pred_wp, _,x_lane,x_junc = self(x, y, tp, light)
                # logits, targets, pred_wp, _= self(x, y, tp, light)
                loss_wp = F.l1_loss(pred_wp, wp)
                _,acc_lane,metrics_lane=self.cri_lane(x_lane,label_lane)
                _,acc_junc,metrics_junc=self.cri_junc(x_junc,label_junc)
                
                self.val_step_outputs.append(metrics_lane)
                self.val_step_outputs_junc.append(metrics_junc)
                self.log(
                    "val/loss_wp",
                    loss_wp,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                
                
                
                self.log(
                    "val/acc_lane",
                    acc_lane,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "val/acc_junc",
                    acc_junc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

        else:
            x, y, wp, tp, light,label_lane,label_junc = batch

            self.y = y
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "val/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )
            self.log(
                "val/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            self.last_epoch = self.current_epoch


    def on_validation_epoch_end(self):
        if self.cfg.model.pre_training.get("multitask", False) == True:
            metrics_lane=self.val_step_outputs[0]
            for item in self.val_step_outputs[1:]:
                metrics_lane+=item
            print('lane:',metrics_lane)
            metrics_junc=self.val_step_outputs_junc[0]
            for item in self.val_step_outputs_junc[1:]:
                metrics_junc+=item
            print('junction:',metrics_junc)
            self.val_step_outputs = []
            self.val_step_outputs_junc = [] 
        
    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )
