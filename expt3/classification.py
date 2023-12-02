"""
@Author: yann
@Date: 2023/12/8
@Mail: yannqu@qq.com
"""
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from typing import Tuple # Before Python 3.9

img_path = "./img/tm7envi.hdr"
gt_path = "./img/groundtruth.tif"


"""
Generate train set and test set in ratio 7:3.
"""
def generate_dataset(gt) -> Tuple[np.ndarray, np.ndarray]:
    (CLASS_1, CLASS_2, CLASS_3, CLASS_4) = 63, 127, 191, 255

    class_arr = [CLASS_1, CLASS_2, CLASS_3, CLASS_4]

    train_ratio = 0.7
    gt_train = np.zeros_like(gt)
    gt_test = np.zeros_like(gt)

    for class_id in range(len(class_arr)):
        rows, cols = np.where(gt == class_arr[class_id])
        perm = np.random.permutation(len(rows))
        for i in range(len(perm)):
            if i / len(perm) <= train_ratio:
                gt_train[rows[perm[i]], cols[perm[i]]] = class_arr[class_id]
            else:
                gt_test[rows[perm[i]], cols[perm[i]]] = class_arr[class_id]

    rows_train, cols_train = np.where(gt_train != 0)
    print("train set size =", len(rows_train))
    rows_test, cols_test = np.where(gt_test != 0)
    print("test set size =", len(rows_test))
    print("total dataset size =", len(rows_train) + len(rows_test))
    return gt_train, gt_test


def precision_evaluation(cla, gt) -> None:  # 精度评定
    def count_number(src) -> Tuple[dict, int]:  # 统计分类数据
        dict_k = {}
        for row in range(src.shape[0]):
            for col in range(src.shape[1]):
                if src[row][col] not in dict_k:
                    dict_k[src[row][col]] = 0
                dict_k[src[row][col]] += 1
        dict_k = dict(sorted(dict_k.items()))
        del dict_k[0]  # 键为0的是未归类的部分,所以去掉
        class_sum = sum(dict_k.values())
        return dict_k, class_sum

    cla_dic, cla_sum = count_number(cla)  # 分类后的
    gt_dic, gt_sum = count_number(gt)  # 真实的
    gt_right = cla * (cla == gt)
    gt_right_dic, gt_right_sum = count_number(gt_right)  # 分类正确的

    p0 = gt_right_sum / gt_sum
    pe = 0

    for gt_key in gt_dic:
        if gt_key not in cla_dic:
            cla_dic[gt_key] = 0
            gt_right_dic[gt_key] = 0
            print("类别%s的用户精度为：0.0000,生产者精度为：0.0000" % gt_key)
        else:
            print("类别%s的用户精度为：%.4f," % (gt_key, gt_right_dic[gt_key] / cla_dic[gt_key]), end='')
            print("生产者精度为：%.4f" % (gt_right_dic[gt_key] / gt_dic[gt_key]))
        pe += gt_dic[gt_key] * cla_dic[gt_key]

    pe = pe / (gt_sum * gt_sum)
    kappa = (p0 - pe) / (1 - pe)
    overall_accuracy = gt_right_sum / gt_sum
    print("-" * 36)
    print("Kappa=", kappa)
    print("overall_accuracy", overall_accuracy)


"""
Unsupervised classification demo.
"""
def unsupervised_demo(img: spy.io.envi.BsqFile) -> None:
    (m, c) = spy.kmeans(img, nclusters=10, max_iterations=30)
    spy.imshow(classes=m)
    plt.figure()
    for i in range(c.shape[0]):
        plt.plot(c[i])
    plt.grid()
    plt.show()


"""
Supervised classification demo.
"""
def supervised_demo(img: spy.io.envi.BsqFile, gt: np.ndarray) -> None:
    gt_train, gt_test = generate_dataset(gt)

    plt.figure()
    classes = spy.create_training_classes(img, gt_train)
    # gmlc = spy.GaussianClassifier(classes)
    gmlc = spy.MahalanobisDistanceClassifier(classes)
    clmap = gmlc.classify_image(img)
    plt.imshow(clmap)
    # spy.imshow(classes=clmap)
    plt.show()

    gt_results = clmap * (gt_test != 0) # Mask
    gt_right = gt_results * (gt_results == gt_test)
    gt_wrong = gt_results * (gt_results != gt_test)
    spy.imshow(classes=gt_right, title="right")
    # plt.imshow(gt_right)
    plt.show()
    spy.imshow(classes=gt_wrong, title="wrong")
    # plt.imshow(gt_wrong)
    plt.show()

    precision_evaluation(gt_results, gt_test)


"""
Main function.
"""
def main() -> None:
    img = envi.open(img_path)
    view = spy.imshow(img)
    print(view)
    plt.show()

    plt.figure()
    gt = plt.imread(gt_path)
    spy.imshow(classes=gt)
    # plt.imshow(gt)
    plt.show()

    # unsupervised_demo(img)
    supervised_demo(img, gt)

main()

