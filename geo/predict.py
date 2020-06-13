from training import LightningFcn
from pytorch_lightning import Trainer
import dataset
import os


exp_path = 'fcn_exp'
version = '3'
cktp = '30'
cktp_path = os.path.join(exp_path, 'lightning_logs', f'version_{version}', 'checkpoints', f'epoch={cktp}.ckpt')


model = LightningFcn.load_from_checkpoint(cktp_path)
trainer = Trainer(gpus=1)
trainer.test(model)