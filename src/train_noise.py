import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np

from unet import UNet
from utils import dice_coeff
from data import dataloaders
from train_util import UtilityModel


class NoiseModel(pl.LightningModule):
    def __init__(
        self,
        util_model,
        depth=5,
        channel_factor=6,
        learning_rate=3e-3,
        pretrained=None,
        noise_coeff=0.001,
        min_scale=1.0,
        max_scale=5.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            "depth",
            "channel_factor",
            "learning_rate",
            "noise_coeff",
            "min_scale",
            "max_scale",
        )
        self.util_model = util_model
        for param in self.util_model.parameters():
            param.requires_grad = False

        self.noise_model = UNet(
            in_channels=3,
            out_channels=1,
            depth=self.hparams.depth,
            cf=self.hparams.channel_factor,
        )

        if pretrained is not None:
            self.noise_model.load_state_dict(pretrained)

        self.normal = torch.distributions.normal.Normal(0, 1)
        self.learning_rate = learning_rate
        self.min_scale = self.hparams.min_scale
        self.max_scale = self.hparams.max_scale
        self.noise_coeff = self.hparams.noise_coeff
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):

        B = torch.sigmoid(self.noise_model(x))

        # sample from normal  distribution
        epsilon = self.normal.sample(B.shape).type_as(B)

        # reparametiation trick
        noise = epsilon * (B * (self.max_scale - self.min_scale) + self.min_scale)

        return noise, B

    def configure_optimizers(self):
        return torch.optim.Adam(self.noise_model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        noise, B = self.forward(images)

        # forward pass through utility model
        self.util_model.eval()
        masks_pred = self.util_model(images + noise)

        loss = self.criterion(masks_pred, masks) - self.noise_coeff * torch.mean(
            B.log()
        )
        self.log("train_loss", loss)
        self.log("mean_B", B.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        noise, B = self.forward(images)

        # forward pass through utility model
        self.util_model.eval()
        masks_pred = self.util_model(images + noise)

        loss = self.criterion(masks_pred, masks) - self.noise_coeff * torch.mean(
            B.log()
        )

        dice = dice_coeff(masks_pred > 0.0, masks)
        return {"val_dice": dice, "val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_dice = torch.stack([x["val_dice"] for x in outs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        self.log_dict({"val_dice": avg_dice, "val_loss": avg_loss})


def main(args):
    pl.seed_everything(0)
    imgs = np.load(args.imgs)
    boxes = np.load(args.boxes, allow_pickle=True)
    masks = np.load(args.masks)

    train_dl, valid_dl, test_dl = dataloaders(imgs, boxes, masks, args.batch_size)

    util_model = UtilityModel.load_from_checkpoint(args.utility_model)

    if args.pretrained is not None:
        # use a pretrained utility model as the initialization for noise model
        args.pretrained = UtilityModel.load_from_checkpoint(
            args.pretrained
        ).model.state_dict()

    model = NoiseModel(
        util_model,
        args.depth,
        args.channel_factor,
        args.learning_rate,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        noise_coeff=args.noise_coeff,
        pretrained=args.pretrained,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
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

    # noise model args
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--channel_factor", default=6, type=int)
    parser.add_argument("--learning_rate", default=3e-3, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--min_scale", default=1.0, type=float)
    parser.add_argument("--max_scale", default=5.0, type=float)
    parser.add_argument("--noise_coeff", default=0.001, type=float)

    # utility model
    parser.add_argument("--utility_model", default="models/utility.ckpt")

    # pretrained noise model
    parser.add_argument("--pretrained", default=None)

    # training args
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--epochs", default=100, type=int)

    args = parser.parse_args()
    main(args)
