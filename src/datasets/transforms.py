import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=512):

    return A.Compose(
        [

            A.Resize(
                image_size,
                image_size,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.VerticalFlip(
                p=0.5,
            ),

            A.Rotate(
                limit=20,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                p=0.5,
            ),

            A.Normalize(),

            ToTensorV2(),

        ]
    )


def get_val_transforms(image_size=512):

    return A.Compose(
        [

            A.Resize(
                image_size,
                image_size,
            ),

            A.Normalize(),

            ToTensorV2(),

        ]
    )