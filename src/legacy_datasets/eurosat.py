import os
import torch
import torchvision.datasets as datasets
import re

def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='.data',
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'EuroSAT', 'train')
        valdir = os.path.join(location, 'EuroSAT', 'val')
        testdir = os.path.join(location, 'EuroSAT', 'test')


        self.train_dataset = datasets.EuroSAT(traindir, split='train', transform=preprocess, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.val_dataset = datasets.EuroSAT(valdir, split='val', transform=preprocess, download=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_dataset = datasets.EuroSAT(testdir, split='test', transform=preprocess, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='.data',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='.data',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)


class EuroSATTest(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='.data',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)
