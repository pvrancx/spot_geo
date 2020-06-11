from training import LightningFcn
from pytorch_lightning import Trainer
from argparse import Namespace

args = Namespace(
    data_path='./data', 
    batch_size=256,
    img_size=(28,28), 
    num_workers=4,
    validation_pct = 0.1,
    class_weights = (float(1),float(1)),
    learning_rate=0.001,
    lr_decay= 0.95
)

model = LightningFcn(args)

exp_path = 'fcn_exp/'

trainer = Trainer(gpus=1, default_root_dir=exp_path, max_epochs=100)    
trainer.fit(model)