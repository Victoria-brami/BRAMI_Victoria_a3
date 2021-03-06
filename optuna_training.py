import os

import pytorch_lightning as pl
from pytorch_lightning import Callback
from optuna.integration import PyTorchLightningPruningCallback
from data import data_transforms
import torch
from torch.optim import Adam, SGD
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from torch.nn.modules import CrossEntropyLoss
from model import Net
from torch.cuda.amp import autocast

import ray
import logging
import optimization.optuna_training_configuration as cfg
import joblib
import optuna

from torch.utils.data.dataset import Dataset
import os

#from xnect_graph import suggest_graph
#from module_graph import GraphNet

train_config = cfg.TrainDatasetConfig()
val_config = cfg.ValDatasetConfig()
net_config = cfg.NetConfig()
exec_config = cfg.ExecutionConfig()
optuna_config = cfg.OptunaConfig()


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        print()
        print('Validation accuracy: {}'.format(trainer.callback_metrics['batch_val_acc']))
        self.metrics.append(trainer.callback_metrics)


def dump_study_callback(study, trial):
    joblib.dump(study, 'study.pkl')


# defines the Lightning model which wraps the Pytorch model and its training/val codes
class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.trial = trial

        # g = suggest_graph(trial)
        # net = GraphNet(g)
        net = Net()
        self.model = net

        if net_config.resume_training:
            checkpoint = torch.load(net_config.init_chkp)
            torch.load_state(net, checkpoint, input_has_module=False, target_has_module=False)

        self.model = net
        self.loss = CrossEntropyLoss(reduction='mean')

    def forward(self, data):
        return self.model.forward(data)

    # defines how the model outputs its loss on a given batch
    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        outputs = self.model.forward(inputs)
        loss_value = self.loss(outputs, labels)
        self.log('Train/train_loss', loss_value, on_epoch=True, on_step=True, prog_bar=True)
        return loss_value

    # defines how the model computes its accuracy on a validation batch
    def validation_step(self, batch, batch_idx):  # the accuracy is the AP50 on the fast val set
        inputs, val_labels = batch
        with autocast(enabled=self.precision == 16):
            val_outputs = self.model.forward(inputs)
        loss_value = self.loss(val_outputs, val_labels)
        val_pred = val_outputs.data.max(1, keepdim=True)[1]
        self.log('Val/val_acc', val_pred.eq(val_labels.data.view_as(val_pred)).cpu().sum(), on_epoch=True, on_step=True, prog_bar=True)
        return {'batch_val_acc': val_pred.eq(val_labels.data.view_as(val_pred)).cpu().sum()}

    # aggregates the batch validations and outputs the metric
    def validation_epoch_end(self, outputs):  # the accuracy is the AP50 on the fast val set
        total_val_loss = torch.mean(torch.tensor([x["batch_val_acc"].float() for x in outputs]))
        print('\n Validation Accuracy: {}'.format(total_val_loss))

    def configure_optimizers(self):
        # Learning rate suggestion
        if optuna_config.suggest_learning_rate is not None:  # choosing lr in the given interval
            chosen_lr = self.trial.suggest_loguniform('learning-rate',
                                                      optuna_config.suggest_learning_rate[0],
                                                      optuna_config.suggest_learning_rate[1])
        else:
            chosen_lr = optuna_config.default_learning_rate

        # Weight decay suggestion
        if optuna_config.suggest_weight_decay is not None:  # choosing wd in the given interval
            chosen_weight_decay = self.trial.suggest_uniform('weight-decay',
                                                                optuna_config.suggest_weight_decay[0],
                                                                optuna_config.suggest_weight_decay[1])
        else:
            chosen_weight_decay = optuna_config.default_weight_decay

        # Optimiser suggestion
        if optuna_config.suggest_optimiser is not None:  # choosing optimiser in the given list
            chosen_optimiser = self.trial.suggest_categorical("optimizer", optuna_config.suggest_optimiser)
            if chosen_optimiser == 'Adam':
                return Adam(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
            elif chosen_optimiser == 'SDG':
                return SGD(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
        else:  # hard-coded default to Adam
            return Adam(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(train_config.data + '/train_images',
                                 transform=data_transforms),
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=2)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_config.data + '/val_images',
                                 transform=data_transforms),
            batch_size=val_config.batch_size,
            shuffle=False,
            num_workers=2)
        return val_loader


# defines the [hyperparameters] -> objective value mapping for Optuna optimisation
def objective(trial):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(exec_config.chkp_folder, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
    )

    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=exec_config.epochs,
        gpus=exec_config.gpus,
        callbacks=[metrics_callback],
        # early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        amp_level='O1',
        precision=16,
        num_sanity_val_steps=exec_config.num_validation_sanity_steps
    )

    model = LightningNet(trial)  # this initialisation depends on the trial argument
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_acc"]  # returns the last epoch's validation loss



if __name__ == "__main__":

    if optuna_config.pruner == 'Hyperband':
        print('Hyperband pruner')
        pruner = optuna.pruners.HyperbandPruner(max_resource=optuna_config.n_iters,
                                                reduction_factor=optuna_config.reduction_factor)
    elif optuna_config.pruner == 'Median':
        print('Median pruner')
        pruner = optuna.pruners.MedianPruner()
    else:
        print('No pruner (or invalid pruner name)')
        pruner = optuna.pruners.NopPruner()

    # initialise the multiprocessing handler
    ray.init(num_cpus=val_config.num_workers, num_gpus=exec_config.gpus, logging_level=logging.CRITICAL,
             ignore_reinit_error=True)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=optuna_config.n_trials, timeout=optuna_config.timeout,
                   callbacks=[dump_study_callback], n_jobs=optuna_config.n_jobs)

    # displays a study summary
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # dumps the study for use with dash_study.py
    joblib.dump(study, 'optimization/study_lr_finished.pkl')
