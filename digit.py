import pandas as pd
import numpy as np
import cv2 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score as AC
import os , time,ssl
from PIL import Image
import PIL.ImageOps

# to setup https context fetch data from openml
if (not os.environ.get("PYTHONHTTPSVERIFY","") and getattr(ssl,"_create_unverified_context",None)) :
    ssl._create_default_https_context = ssl._create_unverified_context
    

# to fetch data from opml lib  
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.array(X)
# to print the values and its count
# print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
n = len(classes)

x_train ,x_test ,y_train ,y_test = tts(X,y,random_state=4,train_size =7500,test_size=2500 )
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

clf = LR(solver="saga",multi_class="multinomial").fit(x_train_scale,y_train)

y_predcit =  clf.predict(x_test_scale)
acc = AC(y_test,y_predcit)
print(acc)

# to start the cammra
cam = cv2.VideoCapture(0)

while True:
    # to capture every frame 
    try :
        ret,frame = cam.read()
        # to draw a rectangle 
        height , width = gray.shape
        upl = (int(width/2-56),int(height/2-56))
        btr = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upl,btr,(0,255,0),2)
        # to find the area and to diect
        roi = gray[upl[1]:btr[1],upl[0]:btr[0]]
        # to convert cv2 img to pil format 
        imagepil = Image.fromarray()
        imagepil_grayscale = imagepil.convert("L")
        imagepil_resize =  imagepil_grayscale.resize((28,28),Image.ANTIALIAS)
        # to invert the img 
        img_in = PIL.ImageOps.invert(imagepil_resize)
        pixel_filter = 20
        #  to convert the image for scaler quntatiy 
        min_pixel =  np.percentile(img_in)
        # to limit the values btw 0 255
        img_lit =  np.clip(img_in-min_pixel,0,255)
        max_pixel =  np.max(img_in)
        # to convert in array 
        img_lit = np.asarray(img_lit)/max_pixel
        test_sample = np.array(img_lit).reshape(1,784)
        test_predict = clf.predict(test_sample)
        print(test_predict)
        #  to display the result 
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) and 0XFF == ord("q"):
            break;
    except Exception as e:
        pass;
        
    
cam.release()
# out.release()
cv2.destroyAllWindows()
    
