"""
@Author: yann
@Date: 2023/12/8
@Mail: yannqu@qq.com
"""

import spectral as spy
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple # Before Python 3.9
from classification import generate_dataset, precision_evaluation, img_path, gt_path

"""
Main function.
"""
def main() -> None:
    plt.figure()
    img = envi.open(img_path)
    view = spy.imshow(img)
    print(view)

    plt.figure()
    gt = plt.imread(gt_path)
    spy.imshow(gt)

    gt_train, gt_test = generate_dataset(gt)

    plt.figure()
    classes = spy.create_training_classes(img, gt_train)
    nfeatures = img.shape[-1]
    nclasses = len(classes)
    print(nclasses)
    p = spy.PerceptronClassifier([nfeatures, 20, 8, nclasses])
    p.train(classes, clip=0., accuracy=100., batch=1,
            momentum=0.3, rate=0.3)
    c = p.classify_image(img)
    plt.imshow(c)

    gt_results = c * (gt_test != 0) # Mask
    gt_right = gt_results * (gt_results == gt_test)
    gt_wrong = gt_results * (gt_results != gt_test)
    spy.imshow(classes=gt_right, title="right by perceptron classifier")
    spy.imshow(classes=gt_wrong, title="wrong by perceptron classifier")
    precision_evaluation(gt_results, gt_test)
    plt.show()

main()