import numpy as np
import torch
import cv2
import time
from PIL import Image
from tqdm import tqdm
from typing import Sequence

from train_noise import NoiseModel
from train_util import UtilityModel
from utils import dice_coeff
from data import dataloaders
from unet import UNet


class GradCAMUNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def activations_hook(self, grads):
        self.gradients = grads

    def forward(self, x):
        # This is the same as the UNet forward,
        # just with added gradient hook
        outs = []
        for i, down in enumerate(self.model.downs):
            x = down(x)
            if i != (self.model.depth - 1):
                outs.append(x)
                x = self.model.max(x)

        self.activations = x.detach()
        x.register_hook(self.activations_hook)

        for i, up in enumerate(self.model.ups):
            x = up(x, outs[-i - 1])

        return self.model.conv1x1(x)


def grad_cam(model, image, x=0, y=0):
    model = GradCAMUNet(model)
    model.eval()
    mask_pred = model(image)
    mask_pred[0, 0, y, x].backward()

    gradients, activations = model.gradients, model.activations

    # global pool
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap


def occlusion_sensitivity(model, images, masks, patch=10, stride=1, n_batches=8):
    """
    Code modified from https://github.com/kazuto1011/grad-cam-pytorch/blob/master/grad_cam.py

    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17

    Originally proposed in:
    Visualizing and Understanding Convolutional Networks
    https://arxiv.org/abs/1311.2901
    """

    def score(pred, target):
        """
        Dice Coefficient
        """
        eps = 1e-10
        num = pred.size(0)
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2).sum(dim=-1)

        return (2.0 * intersection) / (m1.sum(dim=-1) + m2.sum(dim=-1) + eps)

    model.eval()
    mean = 0.0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline dice coefficient without occlusion
    baseline = score(model(images), masks)

    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches)):
        batch_images = []
        batch_masks = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_masks.append(masks)
        batch_images = torch.cat(batch_images, dim=0)
        batch_masks = torch.cat(batch_masks, dim=0)
        scores = score(model(batch_images), batch_masks)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)
    return diffmaps


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

UTILITY_MODEL = "models/utility.ckpt"

LARGE_NOISE_MODEL = "models/unoise_large_pretrained.ckpt"
SMALL_NOISE_MODEL = "models/unoise_small.ckpt"

IMAGES = "data/images.npy"
BOXES = "data/bounding_boxes.npy"
MASKS = "data/masks.npy"
BATCH_SIZE = 32
NUM_IMAGES = 5
SAVE = True


def impose(img, heatmap=None, mask=None):
    img = np.uint8((img.permute(1, 2, 0).detach().cpu().numpy() * STD + MEAN) * 255)
    if heatmap is not None:
        heatmap = cv2.resize(heatmap.cpu().numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return np.uint8(heatmap * 0.4 + img * 0.6)
    if mask is not None:
        mask = (mask.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(int)
        mask = np.concatenate([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        return np.uint8(mask * 0.4 + img * 0.6)
    return img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    imgs = np.load(IMAGES)
    boxes = np.load(BOXES, allow_pickle=True)
    masks = np.load(MASKS)

    # only care about validation dataloader
    _, valid_dl, test_dl = dataloaders(imgs, boxes, masks, BATCH_SIZE)
    ds = valid_dl.dataset

    util_model = UtilityModel.load_from_checkpoint(UTILITY_MODEL)
    model = util_model.model.to(device)

    np.random.seed(42)
    choices = np.random.choice(np.arange(len(ds)), NUM_IMAGES)

    grad_cam_time = 0.0
    unoise_time = 0.0
    occlusion_sensitivity_time = 0.0

    for i, choice in enumerate(choices):
        util_model = UtilityModel.load_from_checkpoint(UTILITY_MODEL)
        model = util_model.model.to(device)

        img, mask = ds[choice]
        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        if i == 0:
            noise_model = NoiseModel.load_from_checkpoint(
                LARGE_NOISE_MODEL, util_model=util_model
            ).to(device)
            B = noise_model(img)[1]
            for threshold in np.linspace(0.0, 1.0, 11):
                thresh_img = img * (B <= threshold)

                thresh_img = Image.fromarray(impose(thresh_img[0]))
                if SAVE:
                    thresh_img.save(f"visualizations/threshold_{threshold:.1f}.png")

            # for some reason these need to be loaded again
            # otherwise gradient hook for grad cam gets messed up
            util_model = UtilityModel.load_from_checkpoint(UTILITY_MODEL)
            model = util_model.model.to(device)

        # plain image
        plain = Image.fromarray(impose(img[0]))
        if SAVE:
            plain.save(f"visualizations/plain_{i}.png")

        # original mask
        og = Image.fromarray(impose(img[0], mask=mask[0]))
        if SAVE:
            og.save(f"visualizations/original_{i}.png")

        # grad cam
        y, x = torch.where(mask[0, 0] > 0)
        start = time.time()
        heatmap = grad_cam(model, img, x[0], y[0])
        grad_cam_time += time.time() - start
        gc = Image.fromarray(impose(img[0], heatmap))
        if SAVE:
            gc.save(f"visualizations/grad_cam_{i}.png")

        # unoise large
        with torch.no_grad():
            noise_model = NoiseModel.load_from_checkpoint(
                LARGE_NOISE_MODEL, util_model=util_model
            ).to(device)
            start = time.time()
            heatmap = -noise_model.noise_model(img)[0, 0]
            heatmap = torch.relu(heatmap) / heatmap.max()
            unoise_time += time.time() - start

        large = Image.fromarray(impose(img[0], heatmap))
        if SAVE:
            large.save(f"visualizations/unoise_large_{i}.png")

        # unoise large, noised image
        # TODO: remove duplications
        with torch.no_grad():
            noise_model = NoiseModel.load_from_checkpoint(
                LARGE_NOISE_MODEL, util_model=util_model
            ).to(device)
            noised = noise_model(img)[0][0]

        noised = Image.fromarray(impose(noised))
        if SAVE:
            noised.save(f"visualizations/noised_{i}.png")

        # unoise small
        with torch.no_grad():
            noise_model = NoiseModel.load_from_checkpoint(
                SMALL_NOISE_MODEL, util_model=util_model
            ).to(device)
            heatmap = -noise_model.noise_model(img)[0, 0]
            heatmap = torch.relu(heatmap) / heatmap.max()

        small = Image.fromarray(impose(img[0], heatmap))
        if SAVE:
            small.save(f"visualizations/unoise_small_{i}.png")

        # occlusion_sensitivity
        with torch.no_grad():
            start = time.time()
            heatmap = occlusion_sensitivity(util_model, img, mask, patch=15, stride=2)[
                0
            ]
            heatmap = heatmap - heatmap.min()
            heatmap /= heatmap.max()
            occlusion_sensitivity_time += time.time() - start
        os = Image.fromarray(impose(img[0], heatmap))
        if SAVE:
            os.save(f"visualizations/occlusion_{i}.png")

    print("unoise_time:", unoise_time / NUM_IMAGES)
    print("grad_cam_time:", grad_cam_time / NUM_IMAGES)
    print("occlusion_sensitivity_time:", occlusion_sensitivity_time / NUM_IMAGES)


if __name__ == "__main__":
    main()
