from typing import Iterable, List, Optional

import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
import pytorch_lightning as pl 
from pytorch_lightning import seed_everything, Trainer
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
from torch.utils.data import DataLoader
from config import cfg
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_helper import OverviewData
from model import CLSModel

if __name__ == '__main__':
    seed_everything(cfg.SEED, workers=True)
    train_df = pd.read_csv(cfg.TRAIN_CSV)

    train_df = train_df[train_df.cat3.isin(['한식','야영장,오토캠핑장','바/까페'])].sample(30,random_state=42)
    mapping = {'바/까페':0, '한식':1, '야영장,오토캠핑장':2}
    train_df['cat3_encode'] = train_df['cat3'].map(mapping)
    tokenizer = AutoTokenizer.from_pretrained(cfg.PRETRAINED_PATH)
    train, valid = train_test_split(train_df, train_size=0.8, random_state=cfg.SEED)

    train_dataset = OverviewData(train, tokenizer)
    valid_dataset = OverviewData(valid, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False,  num_workers=12)

    cfg.N_TRAINING_DATA = len(train_dataset)
    # model = AutoModel.from_pretrained(cfg.PRETRAINED_PATH)
    # output = model(train_dataset[:]['input_ids'], train_dataset[:]['attention_mask'])
    # print(output)


    tb_logger = TensorBoardLogger('logs/')
    checkpoint_callback = ModelCheckpoint(
            dirpath='models',
            save_top_k=1,
            monitor='valid f1-score',
            mode='max',
            save_weights_only=True,
            filename='{epoch}',
        )
    model = CLSModel(cfg)
    trainer = Trainer(
            logger=tb_logger,
            accelerator='cpu',
            callbacks=[checkpoint_callback],
            gradient_clip_val=1,
            accumulate_grad_batches=cfg.ACCUMULATE_GRAD_BATCHES,
            max_epochs=cfg.NUM_TRAIN_EPOCHS,
            precision=16,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )
        
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader])

