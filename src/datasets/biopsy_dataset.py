from torch.utils.data import Dataset


class BiopsyDataset(Dataset):

    def __init__(self, image_dir, mask_dir=None, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = []

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        raise NotImplementedError(
            "Dataset parsing will be implemented when data format is known."
        )