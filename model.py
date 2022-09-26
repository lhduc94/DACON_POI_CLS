from cProfile import label
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import pytorch_lightning as pl

class CLSModel(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASSES
        self.model = AutoModel.from_pretrained(cfg.PRETRAINED_PATH)
        self.hidden_size = self.model.config.hidden_size
        self.dropout_prob = self.model.config.hidden_dropout_prob
        self.classifier = nn.Sequential(
                            nn.Linear(self.hidden_size, cfg.NUM_CLASSES))
        # self.init_weights()
        print('--Init--')
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        print('forward')
        output = self.model(input_ids, attention_mask)
        print('output')
        logits = self.classifier(output.pooler_output)
        print('logits')
        loss = 0
        if labels is not None:
            loss = self.calc_loss(logits, labels)
        print(loss)
        return loss, logits

    def calc_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss
            
    def training_step(self, train_batch, batch_idx):
        input_ids =  train_batch["input_ids"]
        attention_mask = train_batch["attention_mask"]
        labels = train_batch['label']

        loss, logits = self(input_ids, attention_mask, labels)
        predictions = logits.argmax(1)
        print('prediction', predictions)
        self.log('train_loss', loss)

        return {"loss": loss, "prediction": predictions, "label": labels}
    
    def validation_step(self, val_batch, batch_idx):
        input_ids =  val_batch["input_ids"]
        attention_mask = val_batch["attention_mask"]
        labels = val_batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)
        predictions = logits.argmax(1)
        self.log('val_loss', loss)

        return  {"loss": loss, "prediction": predictions, "label": labels}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_label in output['label'].detach().cpu():
                labels.append(out_label)
            for out_prediction in output['prediction'].detach().cpu():
                predictions.append(out_prediction)
        labels = torch.stack(labels)
        predictions = torch.stack(predictions)
        print('label', labels)
        print('prediction',predictions)
        score = f1_score(labels, predictions, average='weighted')
        print(score)
        self.log('train f1-score', score)
    def validation_epoch_end(self, outputs):
        labels = []
        predictions = []
        print('output',outputs)
        for output in outputs:
            for out_label in output['label'].detach().cpu():
                labels.append(out_label)
            for out_prediction in output['prediction'].detach().cpu():
                predictions.append(out_prediction)
        labels = torch.stack(labels)
        predictions = torch.stack(predictions)
        print('label', labels)
        print('prediction',predictions)
        score = f1_score(labels, predictions, average='weighted')
        
        self.log('valid f1-score', score)
    
    def configure_optimizers(self):

        total_step = self.cfg.N_TRAINING_DATA // self.cfg.ACCUMULATE_GRAD_BATCHES * self.cfg.NUM_TRAIN_EPOCHS
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.WEIGHT_DECAY,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.LEARNING_RATE, 
                          eps=self.cfg.ADAM_EPSILON)
        scheduler =  get_linear_schedule_with_warmup(optimizer, 
                                                     num_warmup_steps=self.cfg.N_WARMUP_STEP,
                                                     num_training_steps=total_step)
        
        return [optimizer], [scheduler]