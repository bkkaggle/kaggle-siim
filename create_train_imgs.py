import pydicom
import cv2
import glob
from tqdm import tqdm
import numpy as np

files = glob.glob("./dicom-images-train/*/*/*.dcm")

[file for file in files]

for file in tqdm(files, total=len(files)):
    name = file.split("/")[4].split(".dcm")[0]
    img = pydicom.dcmread(file).pixel_array
    cv2.imwrite(f"./train_imgs/{name}.png", img)
