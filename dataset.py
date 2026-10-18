import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



class ICLEVRTrainDataset(Dataset):
    def __init__(
        self,
        image_dir,
        train_json_path,
        object_json_path,
        image_size=64,
        return_filename=False
    ):
        self.image_dir = image_dir
        self.return_filename = return_filename

        with open(train_json_path, "r") as f:
            self.train_data = json.load(f)

        with open(object_json_path, "r") as f:
            self.object_map = json.load(f)

        self.filenames = list(self.train_data.keys())
        self.num_classes = len(self.object_map)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = self.train_data[filename]
        label_vector = self.labels_to_multihot(labels)

        if self.return_filename:
            return image, label_vector, filename, labels

        return image, label_vector

    def labels_to_multihot(self, labels):
        y = torch.zeros(self.num_classes, dtype=torch.float32)

        for label in labels:
            if label not in self.object_map:
                raise ValueError(f"Unknown label: {label}")

            idx = self.object_map[label]
            y[idx] = 1.0

        return y




class ICLEVRConditionDataset(Dataset):
    """
    For test.json and new_test.json.

    These files do not contain images.
    They only contain label conditions.
    """
    def __init__(self, condition_json_path, object_json_path):
        with open(condition_json_path, "r") as f:
            self.conditions = json.load(f)

        with open(object_json_path, "r") as f:
            self.object_map = json.load(f)

        self.num_classes = len(self.object_map)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        labels = self.conditions[idx]
        label_vector = self.labels_to_multihot(labels)

        return label_vector, labels

    def labels_to_multihot(self, labels):
        y = torch.zeros(self.num_classes, dtype=torch.float32)

        for label in labels:
            if label not in self.object_map:
                raise ValueError(f"Unknown label: {label}")

            idx = self.object_map[label]
            y[idx] = 1.0

        return y





def multihot_to_labels(multihot,object_map):

    """
    Convert a multi-hot vector back to object names.
    Useful for debugging.
    """
    idx_to_object = {idx:name for name ,idx in object_map.items()}

    labels = []

    for idx,value in enumerate(multihot):
        if value >0.5:
            labels.append(idx_to_object[idx])

    return labels



    from torch.utils.data import DataLoader


