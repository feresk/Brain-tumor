

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

NO_PATH = r'archive\brain_tumor_dataset\no'
YES_PATH = r'archive\brain_tumor_dataset\yes'



#print(onlyfiles)


def DataframeFromDir(path, classification):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    rows = []
    for filename in onlyfiles :
        img_path = join(path,filename)

        img = Image.open(img_path).resize((200, 200))
        imgInv = ImageOps.mirror(img)
        for i in range(0,271,90) :
            img = img.rotate(i)
            bw = img.convert('L')
            imgArray = np.array(bw)
            rows.append({'ImageArray':imgArray.ravel(), 'Category':classification})

        for i in range(0,271,90) :
            imgInv = imgInv.rotate(i)
            bw = imgInv.convert('L')
            imgArray = np.array(bw)
            rows.append({'ImageArray':imgArray.ravel(), 'Category':classification})

    return pd.DataFrame(rows)

data = pd.concat([DataframeFromDir(NO_PATH, 0),
                DataframeFromDir(YES_PATH, 1)], ignore_index=True)
#MOMKEN ZEYDA
#data = pd.concat([data.sample(frac=1)], ignore_index=True)

print(data)

features = data.ImageArray
result = data.Category

xTrain, xTest, yTrain, yTest = train_test_split (features, result, test_size=.2,
                random_state=13)

np.savetxt('brain_tumor_xtrain.csv', [i for i in xTrain], delimiter=",", fmt='%d')
np.savetxt('brain_tumor_xtest.csv', [i for i in xTest], delimiter=",", fmt='%d')
np.savetxt('brain_tumor_ytrain.csv', [i for i in yTrain], delimiter=",", fmt='%d')
np.savetxt('brain_tumor_ytest.csv', [i for i in yTest], delimiter=",", fmt='%d')
