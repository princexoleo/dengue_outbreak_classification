import datetime as datetime
from django.contrib.messages import api
from django.http import HttpResponse
from django.shortcuts import render
import requests
from datetime import timezone, datetime
from datetime import datetime
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from numpy import nan
from numpy import isnan
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import cv2
from PIL import Image, ImageDraw, ImageFont

###
import albumentations
import matplotlib.pyplot as plt
import pandas as pd

import tez
from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision

from sklearn import metrics, model_selection, preprocessing


#####
def data_to_image(data):
    """
     take argument data as numpy array
     return a image of numpy array
    """
    data_images = []
    font = ImageFont.truetype("arial.ttf", size=50)
    for dat in data:
        background = np.array([[0 for _ in range(255)] for _ in range(255)], dtype='uint8')
        image = Image.fromarray(background)
        draw = ImageDraw.Draw(image)
        draw.text((32, 32), str(dat[0]), fill='white', font=font)
        draw.text((32, 160), str(dat[1]), fill='white', font=font)
        draw.text((160, 32), str(dat[2]), fill='white', font=font)
        draw.text((160, 160), str(dat[3]), fill='white', font=font)
        rgb = [np.array(image, dtype='uint8') for _ in range(3)]
        data_images.append(rgb)

    return np.array(data_images) / 255


# Create your views here.
def home(request):
    date = request.GET.get('date')
    location = request.GET.get('location')
    ndate = '18/09/20'
    if date is None:
        date = '18/09/20'
    else:
        ndate = str(date)

    if location is None:
        location = "dhaka"

    stm = datetime.strptime(ndate, '%y/%m/%d')
    mdate = datetime.timestamp(stm)
    mdate = int(mdate)
    print(mdate)
    key = 'e60caa1ed3a745218bbb9d045c623c39'
    geocoder = OpenCageGeocode(key)
    results = geocoder.geocode(location)
    lat = results[0]['geometry']['lat']

    lng = results[0]['geometry']['lng']
    print(lat, lng)
    lat = str(lat)
    lng = str(lng)
    print(ndate)

    ##url = 'http://api.openweathermap.org/data/2.5/weather?q={}&type=daily&start={}&end={}&appid=4cca3e952015aecd3cfff6f3c52ccd4b'
    apkey = "774d03c88dcaf477b6abab6949febe7f"
    ##url="https://api.darksky.net/forecast/"+apkey+lat+","+lng+","+ndate
    url = "https://api.darksky.net/forecast/774d03c88dcaf477b6abab6949febe7f/{},{},{}"

    r = requests.get(url.format(lat, lng, mdate)).json()
    print(r)

    features = {
        'City': location,
        'Latitude': lat,
        'Longitude': lng,
        'moonPhase': 0.89,
        'PrecipIntensity': 0.0001,
        'precipIntensityMax': 0.0006,
        'precipProbability': 0.13,
        'temperatureHigh': 94.5,
        'temperatureLow': 79.77,
        'apparentTemperatureHigh': 111.09,
        'apparentTemperatureLow':89.8,
        'dewPoint': 78.28,
        'humidity': 0.77,
        'pressure': 1002.5,
        'windSpeed': 9,
        'windGust': 9,
        'windBearing': 9,
        'cloudCover': 9,
        'uvIndex': 9,
        'ozone': 9,
        'temperatureMin': 9,
        'temperatureMax': 9,
        'visibility': 9,
        'apparentTemperatureMin': 9,
        'apparentTemperatureMax': 9,

    }

    context = {'f': features}
    data = list(features.items())


    #########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inp = torch.rand((1, 25))
    # input_image =  marray = np.array(data)
    #print(marray.shape)
    input_image = data_to_image(inp)
    print(input_image.shape)
    model = DengueModel(num_classes=3)

    model.load("dengue_model_v_2.pth")


    res = model(input_image)
    print(res)
    ###

    ###
    return render(request, "home/home.html", context)


class DengueModel(tez.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.convnet = torchvision.models.resnet18(pretrained=True)
        self.convnet.fc = nn.Linear(512, num_classes)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=0.7)
        return sch

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        outputs = self.convnet(image)

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None
