import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from numpy.random import random
import os


class PlateNumberDataset(Dataset):

    def __init__(self, images_dir, landmarks_path, transform=None):
        self.image_paths = []
        for i in range(3000):
            path = os.path.join(images_dir, '{:0>4d}.jpg'.format(i))
            if os.path.exists(path):
                self.image_paths.append(path)

        landmarks = []
        with open(landmarks_path) as f:
            for line in f.readlines():
                landmark = map(float, line.strip().split()[2:])
                landmark = list(landmark)
                landmarks.append(landmark)
        self.landmarks = np.array(landmarks, dtype=np.float32)

        print('init...')

        for idx in range(self.landmarks.shape[0]):
            img = cv2.imread(self.image_paths[idx])
            h, w, _ = img.shape
            self.landmarks[idx][[0, 2, 4, 6]] /= w
            self.landmarks[idx][[1, 3, 5, 7]] /= h

        self.transform = transform

    def __len__(self):
        return self.landmarks.shape[0]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        img = img[:, :, [2, 1, 0]]
        landmark = self.landmarks[idx]

        sample = {'image': img, 'landmark': landmark}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class Rescale():

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        height, width, _ = image.shape
        # landmark[[0, 2, 4, 6]] /= width
        # landmark[[1, 3, 5, 7]] /= height
        image = cv2.resize(sample['image'], self.size)

        return {'image': image, 'landmark': landmark}


class RandomCrop():

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        l = landmark[[0, 2, 4, 6]].min()
        l = l if l > 0.0 else 0.0
        r = landmark[[0, 2, 4, 6]].max()
        r = r if r < 1.0 else 1.0
        up = landmark[[1, 3, 5, 7]].min()
        up = up if up > 0.0 else 0.0
        down = landmark[[1, 3, 5, 7]].max()
        down = down if down < 1.0 else 1.0

        h, w, _ = image.shape

        l = (l / 2.0) * random()
        r = (1.0 - r) / 2.0 * random() + r
        up = (up / 2.0) * random()
        down = (1.0 - down) / 2.0 * random() + down

        landmark[[0, 2, 4, 6]] = (landmark[[0, 2, 4, 6]] - l) / (r - l)
        landmark[[1, 3, 5, 7]] = (landmark[[1, 3, 5, 7]] - up) / (down - up)

        l = int(l * w)
        r = int(r * w)
        up = int(up * h)
        down = int(down * h)

        image = image[up:down, l:r]
        image = cv2.resize(image, self.size)

        return {'image': image, 'landmark': landmark}


class ToTensor():

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        image = transforms.ToTensor()(image)
        landmark = torch.from_numpy(landmark)
        return {'image': image, 'landmark': landmark}


class Normalize():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        for c in range(3):
            image[:, c] = (image[:, c] - self.mean[c]) / self.std[c]
        return {'image': image, 'landmark': landmark}


if __name__ == '__main__':
    images_dir = '/home/deng/Documents/recognization/data/train'
    landmarks_path = '/home/deng/Documents/recognization/data/train.txt'

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    transform = transforms.Compose([Rescale(224), ToTensor(), Normalize(mean, std)])
    dataset = PlateNumberDataset(images_dir, landmarks_path, transform)
    loader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, num_workers=0)

    for j in range(5):
        for batch in loader:
            print(batch['landmark'])

            img = batch['image'][0]
            # landmark = batch['landmark'][0]
            h, w, _ = img.shape
            print(j)
