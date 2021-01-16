import pytorch_lightning as pl
import torch
import numpy as np
from argparse import ArgumentParser

from unet import UNet
from utils import dice_coeff
from data import dataloaders


class UtilityModel(pl.LightningModule):
    def __init__(self, depth=5, channel_factor=6, learning_rate=3e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(depth=self.hparams.depth, cf=self.hparams.channel_factor)
        self.learning_rate = self.hparams.learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks_pred = self.model(images)
        loss = self.criterion(masks_pred, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks_pred = self.model(images)
        loss = self.criterion(masks_pred, masks)
        dice = dice_coeff(masks_pred > 0.0, masks)
        return {"val_loss": loss, "val_dice": dice}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        avg_dice = torch.stack([x["val_dice"] for x in outs]).mean()
        self.log_dict({"val_loss": avg_loss, "val_dice": avg_dice})


def main(args):
    imgs = np.load(args.imgs)
    boxes = np.load(args.boxes, allow_pickle=True)
    masks = np.load(args.masks)

    train_dl, valid_dl, test_dl = dataloaders(imgs, boxes, masks, args.batch_size)

    model = UtilityModel(args.depth, args.channel_factor, args.learning_rate)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor="val_dice", mode="max")
    trainer = pl.Trainer(
        gpus=args.gpus, max_epochs=args.epochs, checkpoint_callback=checkpoint_cb
    )

    trainer.fit(model, train_dl, valid_dl)


if __name__ == "__main__":
    parser = ArgumentParser()

    # data args
    parser.add_argument("--imgs", default="data/images.npy")
    parser.add_argument("--boxes", default="data/bounding_boxes.npy")
    parser.add_argument("--masks", default="data/masks.npy")

    # model args
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--channel_factor", default=6, type=int)
    parser.add_argument("--learning_rate", default=3e-3, type=float)
    parser.add_argument("--batch_size", default=8, type=int)

    # training args
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--epochs", default=100, type=int)

    args = parser.parse_args()
    main(args)
