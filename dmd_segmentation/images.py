import numpy as np

from os import listdir
from os.path import isfile, join
from PIL import Image as pim


class Image:
    def __init__(self):
        self.resolution = (None, None)

    def get_images(self, foldername):
        files = sorted([ f for f in listdir(foldername) if isfile(join(foldername, f)) ])
        for file in files:
            im = pim.open(join(foldername, file)).convert('L')
            if self.resolution[0] is None:
                self.resolution = (im.size[1], im.size[0])
            yield np.asarray(im)
