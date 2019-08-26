import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model, fold):
    torch.save(model.state_dict(), f"./checkpoint-{fold}.pt")


# From: https://www.kaggle.com/soulmachine/siim-deeplabv3
def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98317
    if lastColor == 255:
        rle.append(runStart)
        rle.append(runLength)
    return " " + " ".join(rle)
