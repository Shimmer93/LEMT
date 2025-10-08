import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

from model.metrics import calulate_error
from misc.lr_scheduler import LinearWarmupCosineAnnealingLR
from misc.utils import torch2numpy, import_with_str, delete_prefix_from_state_dict, exists_and_is_true
from misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP
from misc.vis import visualize_sample
from loss.unsup import UnsupLoss

def create_model(model_name, model_params):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('model', model_name)
    model = model_class(**model_params)
    return model

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    if loss_name == 'UnsupLoss':
        loss_class = UnsupLoss
        loss = loss_class(**loss_params)
    else:
        loss_class = import_with_str('torch.nn', loss_name)
        loss = loss_class(**loss_params)
    return loss

def create_optimizer(optim_name, optim_params, mparams):
    if optim_params is None:
        optim_params = {}
    optim_class = import_with_str('torch.optim', optim_name)
    optimizer = optim_class(mparams, **optim_params)
    return optimizer
    
def create_scheduler(sched_name, sched_params, optimizer):
    if sched_params is None:
        sched_params = {}
    if sched_name == 'LinearWarmupCosineAnnealingLR':
        sched_class = LinearWarmupCosineAnnealingLR
    else:
        sched_class = import_with_str('torch.optim.lr_scheduler', sched_name)
    scheduler = sched_class(optimizer, **sched_params)
    return scheduler

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = create_model(self.hparams.model_name, self.hparams.model_params)

        if exists_and_is_true(self.hparams, 'lemt'):
            self.model_teacher = create_model(self.hparams.model_name, self.hparams.model_params)

        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss_fn = create_loss(self.hparams.loss_name, self.hparams.loss_params)

    def _recover_data(self, data, centroid, radius):
        # print(data.shape, centroid.shape, radius.shape)
        data[..., :3] = data[..., :3] * radius.unsqueeze(-2).unsqueeze(-2) + centroid.unsqueeze(-2).unsqueeze(-2)
        data = torch2numpy(data)
        return data
    
    def _recover_all(self, x, y, y_hat, c, r):
        x = self._recover_data(x.clone().detach(), c, r)
        y = self._recover_data(y.clone().detach(), c, r)
        y_hat = self._recover_data(y_hat.clone().detach(), c, r)
        return x, y, y_hat

    def _calculate_loss(self, batch):

        if exists_and_is_true(self.hparams, 'lemt_pl'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_unsup = batch['point_clouds_unsup']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_lidar = self.model(x_lidar)
            loss_sup = F.mse_loss(y_hat_sup, y_sup) + F.mse_loss(y_hat_lidar, y_lidar)
            
            with torch.no_grad():
                y_hat_unsup0 = self.model(x_unsup[:, :-1])
                y_hat_unsup1 = self.model(x_unsup[:, 1:])
            loss_dynamic, loss_static = self.loss_fn(x_unsup, y_hat_unsup0, y_hat_unsup1)

            loss = loss_sup + self.hparams.unsup_weight * (loss_dynamic + loss_static)
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_unsup': (loss_dynamic + loss_static).item(), 'loss': loss.item()}
            y_hat = y_hat_sup

        elif exists_and_is_true(self.hparams, 'lemt_train'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_pl, y_pl = batch['point_clouds_pl'], batch['keypoints_pl']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_pl = self.model(x_pl)
            y_hat_lidar = self.model(x_lidar)
            loss_sup = F.mse_loss(y_hat_sup, y_sup)
            loss_pl = F.mse_loss(y_hat_pl, y_pl)
            loss_lidar = F.mse_loss(y_hat_lidar, y_lidar)

            loss = loss_sup + self.hparams.pl_weight * loss_pl + self.hparams.lidar_weight * loss_lidar
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_pl': loss_pl.item(), 'loss_lidar': loss_lidar.item(), 'loss': loss.item()}
            y_hat = y_hat_sup
        
        elif exists_and_is_true(self.hparams, 'lemt'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_unsup = batch['point_clouds_unsup']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_lidar = self.model(x_lidar)

            yt_hat_sup = self.model_teacher(x_sup)
            yt_hat_lidar = self.model_teacher(x_lidar)

            loss_sup = F.mse_loss(y_hat_sup, y_sup) + F.mse_loss(yt_hat_sup, y_sup)
            loss_lidar = F.mse_loss(y_hat_lidar, y_lidar) + F.mse_loss(yt_hat_lidar, y_lidar)

            x_unsup0 = x_unsup[:, :-1]
            x_unsup1 = x_unsup[:, 1:]

            with torch.no_grad():
                yt_hat_unsup0 = self.model_teacher(x_unsup0)
                yt_hat_unsup1 = self.model_teacher(x_unsup1)

            y_hat_unsup0 = self.model(x_unsup0)
            y_hat_unsup1 = self.model(x_unsup1)

            loss_pseudo = F.mse_loss(y_hat_unsup0, yt_hat_unsup0.detach()) + F.mse_loss(y_hat_unsup1, yt_hat_unsup1.detach())
            loss_dynamic, loss_static = self.loss_fn(x_unsup, y_hat_unsup0, y_hat_unsup1)

            loss = loss_sup + self.hparams.w_lidar * loss_lidar + \
                              self.hparams.w_pseudo * loss_pseudo + \
                              self.hparams.w_dynamic * loss_dynamic + \
                              self.hparams.w_static * loss_static
            
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_lidar': loss_lidar.item(), 
                         'loss_pseudo': loss_pseudo.item(), 'loss_dynamic': loss_dynamic.item(), 
                         'loss_static': loss_static.item(), 'loss': loss.item()}
            y_hat = y_hat_sup

        else:
            x, y = batch['point_clouds'], batch['keypoints']
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss_dict = {'loss': loss.item()}

        return loss, loss_dict, y_hat
    
    def _visualize(self, x, y, y_hat):
        sample = x[0][0][:, [0, 2, 1]], y[0][0][:, [0, 2, 1]], y_hat[0][0][:, [0, 2, 1]]
        fig = visualize_sample(sample, edges=ITOPSkeleton.bones, point_size=2, joint_size=25, linewidth=2, padding=0.1)
        tb = self.logger.experiment
        tb.add_figure('val_sample', fig, global_step=self.global_step)
        plt.close(fig)
        plt.clf()

    def training_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        loss, loss_dict, y_hat = self._calculate_loss(batch)

        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
        log_dict = {**log_dict, 'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        y_hat = self.model(x)

        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
        log_dict = {'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 10:
            self._visualize(x_rec, y_rec, y_hat_rec)

    def test_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        y_hat = self.model(x)

        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
        log_dict = {'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 10:
            self._visualize(x_rec, y_rec, y_hat_rec)

    def predict_step(self, batch, batch_idx):
        x, c, r, si, gi = batch['point_clouds'], batch['centroid'], batch['radius'], batch['sequence_index'], batch['global_index']
        y_hat = self.model(x)
        y_hat = self._recover_data(y_hat, c, r)
        return y_hat, si, gi

    def configure_optimizers(self):
        if exists_and_is_true(self.hparams, 'lemt'):
            all_params = list(self.model.parameters()) + list(self.model_teacher.parameters())
        else:
            all_params = self.model.parameters()
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, all_params)
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]