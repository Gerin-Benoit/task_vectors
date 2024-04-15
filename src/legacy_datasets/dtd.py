import os
import torch
import torchvision.datasets as datasets


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('.data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'dtd', 'train')
        valdir = os.path.join(location, 'dtd', 'val')
        testdir = os.path.join(location, 'dtd', 'test')
        os.makedirs(traindir, exist_ok=True)
        os.makedirs(valdir, exist_ok=True)
        os.makedirs(testdir, exist_ok=True)

        self.train_dataset = datasets.DTD(
            traindir, split='train', transform=preprocess, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.val_dataset = datasets.DTD(
            valdir, split='val', transform=preprocess, download=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_dataset = datasets.DTD(
            testdir, split='test', transform=preprocess, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]