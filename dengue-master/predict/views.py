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
from torchvision import transforms

from sklearn import metrics, model_selection, preprocessing


#####
import cv2


def text2img(arr, font_size=50, resolution=(512, 512), font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = resolution
    n_colums, n_features = 2, len(arr)
    n_lines = n_features % n_colums + int(n_features / n_colums)
    frame = np.ones((*resolution, 3), np.uint8) * 0

    k = 0
    for i in range(n_colums):
        for j in range(n_lines):
            try:
                cv2.putText(
                    frame, str(arr[k]), (30 + i * (x // n_colums), 5 + (j + 1) * (y // (n_lines + 1))),
                    fontFace=font, fontScale=1, color=(255, 255, 255), thickness=2)
                k += 1
            except IndexError:
                break

    return np.array(frame, np.uint8)


# Create your views here.
def home(request):
    date = request.GET.get('date')
    location = request.GET.get('location')
    ndate = '2020-12-01'
    if date is None:
        date = '2020-12-01'
    else:
        ndate = str(date)

    if location is None:
        location = "dhaka"

    #stm = datetime.strptime(ndate, '%y/%m/%d')
    #mdate = datetime.timestamp(stm)
    #mdate = int(mdate)
    #print(mdate)
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
    #url = "https://api.darksky.net/forecast/774d03c88dcaf477b6abab6949febe7f/{},{},{}"
    #print(url)
    #url ="https://api.darksky.net/forecast/774d03c88dcaf477b6abab6949febe7f/{},{},{}T05:00:00".format(lat,lng,mdate)


    #print(url)
    #r = requests.get(url).json()

    leo_url = "https://api.darksky.net/forecast/9945eb728b3e7a089f4f8f061015ead6/{},{},{}T05:00:00".format(lat,lng,ndate)
    print(leo_url)

    r= requests.get(leo_url).json()
    print(r)

    features = {
        'City': location,
        'Latitude': lat,
        'Longitude': lng,
        'moonPhase': r['daily']['data'][0]['moonPhase'],
        'PrecipIntensity': r['daily']['data'][0]['precipIntensity'],
        'precipIntensityMax': r['daily']['data'][0]['precipIntensityMax'],
        'precipProbability': r['daily']['data'][0]['precipProbability'],
        'temperatureHigh': r['daily']['data'][0]['temperatureHigh'],
        'temperatureLow': r['daily']['data'][0]['temperatureLow'],
        'apparentTemperatureHigh': r['daily']['data'][0]['apparentTemperatureHigh'],
        'apparentTemperatureLow': r['daily']['data'][0]['apparentTemperatureLow'],
        'dewPoint': r['daily']['data'][0]['dewPoint'],
        'humidity': r['daily']['data'][0]['humidity'],
        'pressure': r['daily']['data'][0]['pressure'],
        'windSpeed': r['daily']['data'][0]['windSpeed'],
        'windGust': r['daily']['data'][0]['windGust'],
        'windBearing': r['daily']['data'][0]['windBearing'],
        'cloudCover': r['daily']['data'][0]['cloudCover'],
        'uvIndex': r['daily']['data'][0]['uvIndex'],
        'ozone': r['daily']['data'][0]['ozone'],
        'temperatureMin': r['daily']['data'][0]['temperatureMin'],
        'temperatureMax': r['daily']['data'][0]['temperatureMax'],
        'visibility': r['daily']['data'][0]['visibility'],
        'apparentTemperatureMin': r['daily']['data'][0]['apparentTemperatureMin'],
        'apparentTemperatureMax': r['daily']['data'][0]['apparentTemperatureMax'],

    }
    val = list(features.values())
    val_np = np.array(val)
    val_np = np.reshape(val_np, (len(val), 1))

    context = {'f': features}
    data = list(features.items())



    #########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #inp = torch.rand((1,25),device='cuda:0')
    #inp.cuda()
    inp = val_np
    # input_image =  marray = np.array(data)
    #print(marray.shape)
    input_image = text2img(inp)

    #### Save image
    from PIL import Image as im
    data = im.fromarray(input_image)
    image_name = 'input_'+ndate+'.png'
    data.save(image_name)




    input_image = torch.from_numpy(input_image).float().to(device)
    #input_image.cuda()
    input_image.to(device)
    print(input_image.shape)
    model = DengueModel(num_classes=3)
    model.to(device)

    model.load("dengue_model_v_2.pth")
    if torch.cuda.is_available():
        model.cuda()
        #inp.cuda()

    print("Yes Cudaaaaaaaaaaaaaaaaa: ", type(inp))

    image = Image.open(image_name)
    print(image.show())
    transform = transforms.Compose([transforms.Resize((255, 255)),
                                    transforms.ToTensor(),
                                    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    model.eval()
    res = model(image)
    p = torch.argmax(res[0])
    print(res, p)
    result = res[0][0][p].cpu().detach().numpy()
    import math

    x = math.ceil(result)
    context["class"] = x


   # print(result_dict)
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
