from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import albumentations
from albumentations.pytorch import ToTensor
import numpy as np


imagenet_normalize = Normalize(  # imagenet mean + standard deviation
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

detection_base = albumentations.Compose([ToTensor()])

detection_augment = albumentations.Compose(
    [
        albumentations.HorizontalFlip(),
        albumentations.OneOf(
            [
                albumentations.RandomContrast(),
                albumentations.RandomGamma(),
                albumentations.RandomBrightness(),
            ],
            p=0.3,
        ),
        albumentations.OneOf(
            [
                albumentations.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                albumentations.GridDistortion(),
                albumentations.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ],
            p=0.3,
        ),
        albumentations.ShiftScaleRotate(),
        detection_base,
    ]
)


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        super(SegmentationDataset, self).__init__()
        if augment:
            self.transforms = detection_augment
        else:
            self.transforms = detection_base
        self.images = np.tile(images[..., None], (1, 1, 1, 3))  # 1 channel to 3 channel
        self.masks = masks

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        mask = (mask > 0).astype(int)

        # augment image and bounding boxes together
        augmented = self.transforms(image=img, mask=mask)

        img = imagenet_normalize(augmented["image"])

        return img, augmented["mask"]


def dataloaders(imgs, boxes, masks, batch_size):

    # mask of images where box exist
    positive = boxes != None

    imgs = imgs[positive]
    masks = masks[positive]

    # NOTE: Do not shuffle; data contains multiple images from same patient
    splits = int(imgs.shape[0] * 0.8), int(imgs.shape[0] * 0.9)

    train_ds = SegmentationDataset(
        imgs[: splits[0]], masks=masks[: splits[0]], augment=True
    )
    valid_ds = SegmentationDataset(
        imgs[splits[0] : splits[1]], masks=masks[splits[0] : splits[1]]
    )
    test_ds = SegmentationDataset(imgs[splits[1] :], masks[splits[1] :])

    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4),
        DataLoader(valid_ds, batch_size, num_workers=4),
        DataLoader(test_ds, batch_size, num_workers=4),
    )
