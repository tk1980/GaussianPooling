#%%
import os
import os.path
import sys

import torchvision.datasets as datasets

#%%
class ImageNetValFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, 
    blacklist=os.path.join(os.path.dirname(os.path.abspath(__file__)),'ILSVRC2014_clsloc_validation_blacklist_files.txt')):
        super(ImageNetValFolder, self).__init__(root, transform=transform)

        if blacklist is not None:
            with open(blacklist, mode='r') as f:
                blacklists = [s.strip() for s in f.readlines()]
            
            self.samples = [x for x in self.samples if os.path.split(x[0])[1] not in blacklists]
