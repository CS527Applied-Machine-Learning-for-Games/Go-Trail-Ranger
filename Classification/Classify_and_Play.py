from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
import torch.nn as nn
from torch.optim import Adam
import time
import pyautogui
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

grayscale = True
height = 256
width = 256
coordinates_value = (271, 173, 829, 470)
game_end = (683, 188, 70, 27)


def preprocess_to_classify(img):
    img_size = (height, width)
    img = resize(np.array(img), img_size)
    img = np.transpose(img)
    img = img.astype('float32')
    return img

def preprocess(img):
    img_size = (height, width)
    img = resize(np.array(img), img_size)
    if grayscale:
         img = img.mean(-1, keepdims=True)
    img = np.transpose(img, (2, 0, 1))
    img = np.transpose(img)
    img = img.astype('float32')
    return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Defining another 2D convolution layer
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4096,1024),
            nn.Linear(1024,512),
            nn.Linear(512,2),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def classify():

    # prediction for training and validation set
    output_train = model(x_train)

    # computing  the prediction
    # print(output_train)
    value, preds_train = torch.max(output_train, 1)
    print(value.item())
    return preds_train.item() if value.item() > 1.5 else -1


#load model
PATH = 'small_rbr_model_2.pt'
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

pyautogui.press('space', interval=2)
while True:
   b = time.time()
   train_img = []

   img  = pyautogui.screenshot(region=coordinates_value)
   train_img.append(preprocess(img))

   train_x = np.array(train_img)
   # defining the target

   # defining in torch format
   train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
   train_x = torch.from_numpy(train_x)

   x_train = train_x

   action = classify()

   if action == 0:
     pyautogui.press('right')
     print('Right')

   elif action == 1:
     pyautogui.press('left')
     print('Left')

   del action
   #time.sleep(0.5)
   #text = pytesseract.image_to_string(pyautogui.screenshot(region=game_end), lang='eng', config='--psm 6')
   #if "00" in text:
       #pyautogui.press('space')

print('Done')