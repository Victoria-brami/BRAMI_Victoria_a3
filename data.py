import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
default_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

def build_augmentation(aug_type, input_size):

    augmentations = [
                    transforms.Resize((input_size, input_size))
                ]

    if 'flip' in aug_type:
        augmentations.append(transforms.RandomVerticalFlip(p=0.5))
    if 'rotate' in aug_type:
        augmentations.append(transforms.RandomRotation(degrees=80))
    if 'erasing' in aug_type:
        augmentations.append(transforms.RandomErasing())
    if 'colors' in aug_type:
        augmentations.append(transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0))

    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    data_transforms = transforms.Compose(augmentations)

    return data_transforms

