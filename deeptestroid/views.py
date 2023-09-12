from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
from django.conf import settings
import json
from xgboost import XGBRegressor
CLASS = ['high', 'low', 'medium', 'very-high', 'very-low']
SS_TYPE = ['login', 'shop', 'map', 'social', 'other']
ROOT_DIR = settings.BASE_DIR


def black(request):
    if request.method == 'POST':
        ss_idx = request.POST.get('type', None)
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        print(filename)
        uploaded_file_url = fs.url(filename)
        model = load_model('mobilenetv3/')
        img = cv2.imread(os.path.join(ROOT_DIR, 'media', filename))
        # image=os.path.join(app.root_path, 'static', 'uploads', filename)
        resize = tf.image.resize(img, (224, 224))
        yhat = model.predict_step(np.expand_dims(resize, 0))
        maxx = -1
        yhat = yhat[0]
        iddx = False
        for i in range(len(yhat)):
            print(yhat[i])
            if yhat[i] > maxx:
                maxx = yhat[i]
                iddx = i
        ss = SS_TYPE[int(ss_idx)]
        cl = CLASS[iddx]
        message = ''

        if (ss == 'login'):
            if (cl in ['very-low', 'low']):
                message = 'View is Great'
            elif (cl == 'medium'):
                message = 'View is Fine'
            else:
                message = 'Reduce Element Will be better'
        elif (ss == 'shop'):
            if (cl == 'medium'):
                message = 'View is Great'
            elif (cl in ['high', 'very-high']):
                message = 'View is Fine'
            else:
                message = "You can Add more element"
        elif (ss == 'map'):
            if (cl == 'low'):
                message = 'View is Great'
            elif (cl in ['very-low', 'medium']):
                message = 'View is Fine'
            else:
                message = 'Reduce Element Will be better'
        elif (ss == 'social'):
            if (cl == 'medium'):
                message = 'View is Great'
            elif (cl in ['low', 'high']):
                message = 'View is Fine'
            elif (cl == 'very-low'):
                message = 'You can add more element'
            else:
                message = 'Reduce Element Will be better'
        elif (ss == 'other'):
            if (cl == 'low'):
                message = 'If the purpose of the view is showing data. then adding more element will good.otherwise fine.'
            elif (cl == 'very-low'):
                message = 'If the View has lots of text then reduce it. adding more element will good.'
            elif (cl == 'medium'):
                message = 'View is fine'
            elif (cl == 'high'):
                message = 'If its a landing page then reducing the element will be better. Otherwise fine'
            else:
                message = 'Reduce Element Will be better'

        print(f'{cl}:{message}')
        return HttpResponse(json.dumps({'class': cl, 'message': message}), content_type="application/json")
    return render(request, 'black.html')


def white(request):
    if request.method == 'POST':
        tool = request.POST.get('tool', None)
        model = XGBRegressor()
        model.load_model('xgbreg.json')
        x = list()
        x.append(int(request.POST['plocs']))
        x.append(int(request.POST['mplocs']))
        x.append(int(request.POST['tlocs']))
        x.append(int(request.POST['mtlocs']))
        x.append(int(request.POST['classes']))
        x.append(int(request.POST['aclasses']))
        x.append(int(request.POST['dclasses']))
        x.append(int(request.POST['mclasses']))
        x.append(int(request.POST['methods']))
        x.append(int(request.POST['amethods']))
        x.append(int(request.POST['dmethods']))
        x.append(int(request.POST['mmethods']))
        x.append(int(request.POST['cmmethods']))

        if tool == 'Espresso':
            x = x+[1, 0, 0, 0]
        elif tool == 'Robolectric':
            x = x+[0, 1, 0, 0]
        elif tool == 'Robotium':
            x = x+[0, 0, 1, 0]
        else:
            x = x+[0, 0, 0, 1]

        n_arr = model.predict(np.array([x], dtype=float))
        res = n_arr[0].tolist()
        print(res)

        def zeroor(x):
            if x < 0:
                return -1
            return round(x, 2)
        context = {
            'TLR': zeroor(res[0]),
            'MTRL': zeroor(res[1]),
            'MRTL': zeroor(res[2]),
            'TMR': zeroor(res[3]),
            'MCR': zeroor(res[4]),
            'MMR': zeroor(res[5]),
            'RFCR': zeroor(res[6]),
            'FCR': zeroor(res[7])
        }
        print(context)
        return HttpResponse(json.dumps(context), content_type="application/json")
    return render(request, 'white.html')
