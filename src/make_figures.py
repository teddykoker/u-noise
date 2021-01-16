import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from utils import dice_coeff
from train_noise import NoiseModel
from train_util import UtilityModel
from data import dataloaders

# if you are into serif
# plt.rcParams["font.family"] = "Times New Roman"


def color_linestyle(name):
    if "Small" in name:
        color = "#7570b3"
    elif "Medium" in name:
        color = "#1b9e77"
    else:
        color = "#d95f02"

    linestyle = "--" if "Pretrained" in name else "-"
    return color, linestyle


def plot_results(results):
    ax = plt.subplot(111)
    for name in results:
        color, linestyle = color_linestyle(name)
        ax.plot(
            results[name]["coverage"],
            results[name]["dice"],
            label=name,
            color=color,
            linestyle=linestyle,
        )
    ax.set_xlabel("Average Percent Image Visible")
    ax.set_ylabel("Utility (Dice Coefficient)")
    ax.legend()
    # Hide the right and top spines
    # Only show ticks on the left and bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.grid()
    plt.savefig("figures/figure1.eps")
    plt.show()

    ax = plt.subplot(111)
    regular = {"params": [], "dice": [], "name": []}
    pretrained = {"params": [], "dice": [], "name": []}
    for name in results:
        if "Pretrained" in name:
            pretrained["params"].append(results[name]["num_params"] / 1000)
            pretrained["dice"].append(results[name]["dice_at_half_coverage"])
            pretrained["name"].append(name.split(" ")[1])  # Small, Medium, Large
        else:
            regular["params"].append(results[name]["num_params"] / 1000)
            regular["dice"].append(results[name]["dice_at_half_coverage"])
            regular["name"].append(name.split(" ")[1])  # Small, Medium, Large

    ax.plot(
        pretrained["params"],
        pretrained["dice"],
        label="Pretrained",
        marker="o",
        color="black",
    )
    ax.plot(
        regular["params"],
        regular["dice"],
        label="Not Pretrained",
        marker="o",
        color="black",
        ls="--",
    )

    for d in regular, pretrained:
        for name, x, y in zip(d["name"], d["params"], d["dice"]):
            ax.annotate(
                name, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )

    ax.set_xlabel("Number of Parameters (Thousands)")
    ax.set_ylabel("Utility at 50% Average Image Visibility (Dice Coefficient)")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.27, 0.40)
    ax.legend()

    # Hide the right and top spines
    # Only show ticks on the left and bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.grid()
    plt.savefig("figures/figure2.eps")
    plt.show()


###############################################################################
# Generate Results
# This only needs to be done once
###############################################################################

UTILITY_MODEL = "models/utility.ckpt"

MODELS = {
    "U-Noise Small": "models/unoise_small.ckpt",
    "U-Noise Medium": "models/unoise_medium.ckpt",
    "U-Noise Large": "models/unoise_large.ckpt",
    "U-Noise Small (Pretrained)": "models/unoise_small_pretrained.ckpt",
    "U-Noise Medium (Pretrained)": "models/unoise_medium_pretrained.ckpt",
    "U-Noise Large (Pretrained)": "models/unoise_large_pretrained.ckpt",
}

IMAGES = "data/images.npy"
BOXES = "data/bounding_boxes.npy"
MASKS = "data/masks.npy"
BATCH_SIZE = 32

NUM_THRESHOLDS = 20
RESULTS = "data/results.pickle"


@torch.no_grad()
def evaluate(dataloader, model, thresholds, device):
    # given data, model, and threshold, what is the utility at that threshold,
    # and how much of the image is covered
    model.eval()
    model.to(device)
    dice = [[] for _ in range(len(thresholds))]
    coverage = [[] for _ in range(len(thresholds))]
    bs = []

    for (images, masks) in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        _, B = model(images)
        bs.append(B)

        for i, threshold in enumerate(thresholds):
            thresh_images = images * (B <= threshold)
            masks_pred = model.util_model(thresh_images)
            dice[i].append(float(dice_coeff(masks_pred > 0.0, masks).item()))
            coverage[i].append(((B <= threshold).sum().float() / B.numel()).item())

    dice = [np.mean(dice[i]) for i in range(len(thresholds))]
    coverage = [np.mean(c) for c in coverage]

    bs = torch.cat(bs)
    median_b = torch.median(bs)
    dice_at_half_coverage = []

    for (images, masks) in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        _, B = model(images)
        images = images * (B <= median_b)
        masks_pred = model.util_model(images)
        dice_at_half_coverage.append(dice_coeff(masks_pred > 0.0, masks).item())

    return dice, coverage, np.mean(dice_at_half_coverage)


def generate_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    imgs = np.load(IMAGES)
    boxes = np.load(BOXES, allow_pickle=True)
    masks = np.load(MASKS)

    data = {}
    thresholds = np.linspace(0.0, 1.0, num=NUM_THRESHOLDS + 1)

    # only care about validation dataloader
    _, valid_dl, _ = dataloaders(imgs, boxes, masks, BATCH_SIZE)

    util_model = UtilityModel.load_from_checkpoint(UTILITY_MODEL)

    for name, path in MODELS.items():
        model = NoiseModel.load_from_checkpoint(path, util_model=util_model)
        data[name] = {}
        data[name]["thresholds"] = thresholds
        data[name]["num_params"] = sum(
            p.numel() for p in model.noise_model.parameters()
        )
        (
            data[name]["dice"],
            data[name]["coverage"],
            data[name]["dice_at_half_coverage"],
        ) = evaluate(valid_dl, model, thresholds, device)
        print(f"done: {name}")

    with open(RESULTS, "wb") as f:
        pickle.dump(data, f)

    return data


def main():
    if Path(RESULTS).exists():
        with open(RESULTS, "rb") as f:
            results = pickle.load(f)
    else:
        results = generate_data()
    plot_results(results)


if __name__ == "__main__":
    main()
