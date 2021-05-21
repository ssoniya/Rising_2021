"""
# @Project : The RISING 2021
# @Description : Create dataset from Chest_Xray and Medical_MNIST datasets
#       Dataset with 4 labels - AbdomenCT, BreastMRI, ChestCT, Abnormal
#		Also, generates csv labels file
# @Author : Soniya Singhal
# @dataset : Kaggle Chest Xray Pneumonia & Medical MNIST datasets
# |-datasets
#       |-chest_xray_small <From BinaryClassification>
#       |-medical_mnist_small <From MulticlassClassification>
#
"""
# IMPORT LIBRARIES
import os
import shutil
import pandas as pd
import numpy as np

# PATH OF DATASET
data_path1 = "../datasets/chest_xray_small"
data_path2 = "../datasets/medical_mnist_small"

data_classes1 = ["NORMAL", "PNEUMONIA"]
data_classes2 = ["AbdomenCT", "BreastMRI"]

data_sets = ["train", "test", "val"]

# DEFINE TRAIN-VAL-TEST DATA DIRECTORIES
output = "../datasets/multilabel_dataset"

# Dataset with 4 labels - AbdomenCT, BreastMRI, ChestCT, Abnormal
labels = []
labels.extend(data_classes2)
labels.append("ChestCT")
labels.append("Abnormal")

L = len(labels)
print("Labels:", labels)

samples = np.zeros(3, dtype=int)

# Copy Dataset1
print("Copying Datasets...")
for idx, dset in enumerate(data_sets):
    for fol in data_classes1:
        folPath = os.path.join(data_path1, dset, fol)
        fyls = os.listdir(folPath)
        num = len(fyls)
        samples[idx] += num
        out_category = "ChestCT"
        if fol == "PNEUMONIA":
            out_category = "ChestCT_Abnormal"
        os.makedirs(os.path.join(output, dset, out_category), exist_ok=True)
        for img in fyls:
            shutil.copy(os.path.join(folPath, img), os.path.join(output, dset, out_category, img))
        print("Copied {0}: {1}!".format(dset, fol))

    # Copy Dataset2

    for fol in data_classes2:
        folPath = os.path.join(data_path2, dset, fol)
        fyls = os.listdir(folPath)
        num = len(fyls)
        samples[idx] += num
        os.makedirs(os.path.join(output, dset, fol), exist_ok=True)
        for img in fyls:
            shutil.copy(os.path.join(folPath, img), os.path.join(output, dset, fol, img))
        print("Copied {0}: {1}!".format(dset, fol))

print("Total Train Samples:", samples[0])
print("Total Test Samples:", samples[1])
print("Total Val Samples:", samples[2])

# WRITE CSV LABELS FILE
ct = 0
for idx, dset in enumerate(data_sets):
    imgs = []
    categories = []
    for fol in os.listdir(os.path.join(output, dset)):
        folPath = os.path.join(output, dset, fol)
        fyls = os.listdir(folPath)
        lab = np.zeros(L, dtype=int)
        if fol == labels[0]:
            lab[0] = 1
            print(fol, lab)
        elif fol == labels[1]:
            lab[1] = 1
            print(fol, lab)
        elif fol == labels[2]:
            lab[2] = 1
            print(fol, lab)
        if labels[3] in fol:
            lab[2] = 1
            lab[3] = 1
            print(fol, lab)
        for img in fyls:
            imgs.append(fol + "/" + img)
            categories.append(lab)
    images = np.asarray(imgs)
    category = np.asarray(categories)
    print("Samples Shapes:", images.shape, category.shape)
    df = pd.DataFrame({"Image": images})

    for idx, label in enumerate(labels):
        df[label] = category[:, idx]

    print(df.head())

    df.to_csv(os.path.join(output, dset, dset + "_labels.csv"), index=False)