from argparse import Namespace
from typing import Tuple, Optional

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
            class_weights: Optional[Tuple[float, float]] = None,
            data_path: str = '.',
            validation_pct: float = 0.1,
            img_size: Tuple[int, int] = (224, 224),
            num_workers: int = 4
    ):
        super(LightningFcn, self).__init__()
        self.hparams = Namespace(
            learning_rate=learning_rate,
            batch_size=batch_size,
            lr_decay=lr_decay,
            class_weights=class_weights,
            data_path=data_path,
            validation_pct=validation_pct,
            img_size=img_size,
            num_workers=num_workers
        )
        self.model = create_fcn_resnet()

        assert 0. <= validation_pct <= 1., 'invalid validation ratio'
        dataset = GeoSetFromFolder(
            root=self.hparams.data_path,
            dataset='train',
            output_size=self.hparams.img_size,
            transform=transforms.ToTensor()
        )
        n_data = len(dataset)
        n_train_data = int((1. - validation_pct) * n_data)
        lengths = [n_train_data, (n_data - n_train_data)]
        self.train_set, self.val_set = torch.utils.data.random_split(dataset, lengths)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.hparams.class_weights))

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, self.hparams.lr_decay)
        return [optim], [scheduler]

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y.long())

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
        tensorboard_logs = {'val_loss': avg_loss}
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
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        test_set = GeoSetFromFolder(
            root=self.hparams.data_path,
            dataset='test',
            transform=transforms.ToTensor(),
            output_size=(480, 640)
        )
        return DataLoader(
            test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

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
