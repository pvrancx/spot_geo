from argparse import Namespace

import torch
import torchvision
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from geo.dataset import GeoSetFromFolder
from geo.models import create_fcn_resnet


class LightningFcn(LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-4,
            batch_size: int = 64,
            lr_decay: float = 0.999,
            validation_pct: float = 0.1
    ):
        super(LightningFcn, self).__init__()
        self.hparams = Namespace(**locals())
        self.model = create_fcn_resnet()

        assert 0. <= validation_pct <= 1., 'invalid validation ratio'
        dataset = GeoSetFromFolder(
            root=self.hparams.data_path,
            dataset='training',
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x))
        )
        n_data = len(dataset)
        lengths = [int((1. - validation_pct) * n_data), int(validation_pct * n_data)]
        self.train_set, self.val_set = torch.utils.data.random_split(dataset, lengths)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, self.hparams.lr_decay)
        return [optim], [scheduler]

    def loss(self, y_hat, y):
        idx = y.long()
        mask = torch.zeros_like(y_hat)
        mask.scatter_(2, idx, 1.)
        if self.on_gpu:
            mask.cuda()
        return nn.NLLLoss2d(y_hat, y)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_psnr': avg_psnr}
        return {'val_loss': avg_loss,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss,
                'log': tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        test_set = GeoSetFromFolder(
            root=self.hparams.data_path,
            dataset='test',
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x))
        )
        return DataLoader(test_set, batch_size=self.hparams.batch_size)

    def on_epoch_end(self) -> None:
        sample_input, _ = next(iter(self.trainer.val_dataloaders[-1]))
        if self.on_gpu:
            sample_input = sample_input.cuda()
        # log sampled images
        sample_imgs = self(sample_input)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.experiment.add_scalar(f'learning_rate', current_lr, self.current_epoch)
