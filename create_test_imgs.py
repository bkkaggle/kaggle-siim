import pydicom
import cv2
import glob
from tqdm import tqdm

files = glob.glob("./dicom-images-test/*/*/*.dcm")

for file in tqdm(files, total=len(files)):
    name = file.split("/")[4].split(".dcm")[0]
    cv2.imwrite(f"./test_imgs/{name}.png", pydicom.dcmread(file).pixel_array)
