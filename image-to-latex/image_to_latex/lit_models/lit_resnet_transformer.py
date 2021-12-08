from pathlib import Path
from typing import List

import torch
from torch._C import device
import torch.nn as nn
from pytorch_lightning import LightningModule

from ..data.utils import Tokenizer
from ..models import ResNetTransformer
from .metrics import EditDistance
from .metrics import ExactMatch


##封装了一层的LightningModulepe
class LitResNetTransformer(LightningModule):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        vocab_file = Path(__file__).resolve().parents[3] / "data" / "vocab1.json"
        # vocab_file='/data/zzengae/jwwang/final_project/vocab.json'
        self.tokenizer = Tokenizer.load(vocab_file)
        self.model = ResNetTransformer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.sos_index,
            eos_index=self.tokenizer.eos_index,
            pad_index=self.tokenizer.pad_index,
            num_classes=len(self.tokenizer),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        self.edit_val = EditDistance(self.tokenizer.ignore_indices)
        self.edit_test = EditDistance(self.tokenizer.ignore_indices)
        self.exac_val=ExactMatch(self.tokenizer.ignore_indices)
        self.exac_test=ExactMatch(self.tokenizer.ignore_indices)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train/loss", loss)

        # decoded = self.tokenizer.decode(targets[0].tolist())  # type: ignore
        # decoded_str = " ".join(decoded)
        # print(decoded)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        #dev集合的loss
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        #dev集合的acc
        preds = self.model.predict(imgs)

        edit_val = self.edit_val(preds, targets)



        decoded = self.tokenizer.decode(targets[0].tolist())  # type: ignore
        # decoded_str = " ".join(decoded)
        print('aaaa:    {}'.format(decoded))


        decoded = self.tokenizer.decode(preds[0].tolist())  # type: ignore
        # decoded_str = " ".join(decoded)
        print('bbbb:    {}'.format(decoded))

        # print('val/edit_distance    :{}'.format(edit_val))
        self.log("val_edit_distance", edit_val)
        print(edit_val)
        
        # print(targets)
        # print(preds)
        exac_val=self.exac_val(preds,targets)
        # print('val/exact_match    :{}'.format(exac_val))
        self.log("val_exact_match",exac_val)

    def test_step(self, batch, batch_idx):
        
        #test集合的acc
        # test手动调用 训练过程中不会调用test_step
        imgs, targets = batch
        preds = self.model.predict(imgs)
        
        edit_val = self.edit_val(preds, targets)



        decoded = self.tokenizer.decode(targets[0].tolist())  # type: ignore
        # decoded_str = " ".join(decoded)
        print('aaaa:    {}'.format(decoded))


        decoded = self.tokenizer.decode(preds[0].tolist())  # type: ignore
        # decoded_str = " ".join(decoded)
        print('bbbb:    {}'.format(decoded))

        # print('val/edit_distance    :{}'.format(edit_val))
        self.log("val_edit_distance", edit_val)
        print(edit_val)
        
        # print(targets)
        # print(preds)
        exac_val=self.exac_val(preds,targets)
        # print('val/exact_match    :{}'.format(exac_val))
        self.log("val_exact_match",exac_val)


    def test_epoch_end(self, test_outputs):
        with open("test_predictions.txt", "w") as f:
            for preds in test_outputs:
                for pred in preds:
                    decoded = self.tokenizer.decode(pred.tolist())
                    decoded.append("\n")
                    decoded_str = " ".join(decoded)
                    f.write(decoded_str)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
