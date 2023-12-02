import spectral
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple # Before Python 3.9
from classification import img_path, gt_path, generate_dataset

gt = plt.imread(gt_path)
gt_train, gt_test = generate_dataset(gt)

print("test")

