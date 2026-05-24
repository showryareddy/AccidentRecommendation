#import require python classes and packages
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


main = tkinter.Tk()
main.title("A CNN-Driven Approach for Injury Type and Severity Detection with Hospital Recommendations for Emergency Response") #designing main screen
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")


#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []

#define global variables
X = []
Y = []
path = "Dataset"
labels = []


def getSeverity(image_path):
    injury_type = "Unable to detect"
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 120])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange (hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        red_area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(red_area)
        if w >= 100:
            injury_type = "Major Severity"
        else:
            injury_type = "Minor Severity"
        cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
    return injury_type, img
    

def getRecommendation(label):
    with open("recommendation/"+label+".txt", "rb") as file:
        data = file.read()
    file.close()
    return data.decode()

def loadDataset():
    text.delete('1.0', END)
    global path, labels, name, X, Y, root, img, getLabel
    #define function to load class labels
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    def getLabel(name):
        index = -1
        for i in range(len(labels)):
            if labels[i] == name:
                index = i
                break
        return index
    text.insert(END, "Labels found in dataset are  "+str(labels)+"\n")
    
    #loop and read all images from dataset
    if os.path.exists('model/X.txt.npy'):#if images already processed then load all images
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:#if not processed then read and process each image
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])#read image
                    img = cv2.resize(img, (64, 64))#resize image
                    X.append(img)#addin images features to training array
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)            
    text.insert(END, "Accident Dataset Loading is Completed \n\n")
    text.insert(END, "Total Images Found in Dataset = "+str(X.shape[0])+"\n") 
    
    
def SampleDisplay():
    text.delete('1.0', END)
    global names, count, Y, y_pos, labels, bars, height
    #visualizing class labels count found in dataset
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (8, 6)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Class Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()


def splitDataset():
    text.delete('1.0', END) 
    global X, indices, Y, X_train, X_test, y_train, y_test
    #preprocess images like shuffling and normalization
    X = X.astype('float32')
    X = X/255 #normalized pixel values between 0 and 1
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle all images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END, "Image Preprocessing, and Normalization is Completed \n")
    text.insert(END,"80% dataset for training = "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for training = "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):  
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")        
    classes = ['Hand', 'Head', 'Leg']
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = classes, yticklabels = classes, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(classes)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
        

def MLModels():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, svm_cls, svm_predict, dt_cls, dt_predict, X_train1, y_train1, X_test1, y_test1    
    #reshaping images data to train with ML algorithms
    y_train1 = np.argmax(y_train, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
    X_train1 = X_train1[0:1000]
    X_test1 = X_test1[0:1000]
    #training svm on accident images features
    svm_cls = svm.SVC()
    svm_cls.fit(X_train1, y_train1)#training SVM on train features
    svm_predict = svm_cls.predict(X_test1)#predicting on test features
    #call this function to calculate accuracy and other metrics
    calculateMetrics("SVM Classifier", svm_predict, y_test1)
        
    #training DecisionTreeClassifier on accident images features
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train1, y_train1)#training DecisionTreeClassifier on train features
    dt_predict = dt_cls.predict(X_test1)#predicting on test features
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Decision Tree Classifier", dt_predict, y_test1)
    
    
def EnsembleModel():
    global X_train1, y_train1, X_test1, y_test1, rf_cls, rf_predict
    #training RandomForestClassifier on accident images features
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train1, y_train1)#training RandomForestClassifier on train features
    rf_predict = rf_cls.predict(X_test1)#predicting on test features
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Ensemble Classifier", rf_predict, y_test1)

def Proposed():
    global cnn_model, X_train, y_train, X_test, y_test, hist, f, y_test1, cnn_predict
    #train CNN algorithm on accident detection image features
    cnn_model = Sequential()
    #defining CNN layer with 32 filters of 3 X # matrix for features filtration
    cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    #max layer to collected relevant filtered features from Previous CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    #defining another CNN layer for further features optimization
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    #collect relevant features
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    #change features dimension to single dimension
    cnn_model.add(Flatten())
    #define output layer
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile, train and load model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test images   
    cnn_predict = cnn_model.predict(X_test)
    cnn_predict = np.argmax(cnn_predict, axis=1)
    #y_test1 = np.argmax(y_test, axis=1)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Proposed CCNN Model", cnn_predict, y_test1)

    
def PerformanceGraph():    
    #comparison graph
    df = pd.DataFrame([['Proposed CCNN','Accuracy',accuracy[0]],['Proposed CCNN','Precision',precision[0]],['Proposed CCNN','Recall',recall[0]],['Proposed CCNN','FSCORE',fscore[0]],
                       ['SVM','Accuracy',accuracy[1]],['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','FSCORE',fscore[1]],
                       ['Decision Tree','Accuracy',accuracy[2]],['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','FSCORE',fscore[2]],
                       ['Ensemble','Accuracy',accuracy[3]],['Ensemble','Precision',precision[3]],['Ensemble','Recall',recall[3]],['Ensemble','FSCORE',fscore[3]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 6))
    plt.title("Performance Evaluation")
    plt.show()

    

def predict():
    text.delete('1.0', END)
    global filename, image, img, im2arr, predict, injury_type, recommendation, labels  
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)#read test image
    img = cv2.resize(image, (64,64))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    predict = cnn_model.predict(img)#perform prediction on test image
    predict = np.argmax(predict)
    injury_type, img = getSeverity(filename)
    img = cv2.resize(img, (400,300))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recommendation = getRecommendation(labels[predict])
    text.insert(END, "Recommendation Details \n")
    text.insert(END, recommendation)
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.putText(img, 'Injury Type : '+injury_type, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    plt.figure(figsize=(8,5))
    plt.imshow(img)
    plt.show()

def Exit():
    main.destroy()
    
# Main Window Designing

font = ('times', 16, 'bold')
title = Label(main, text='A CNN-Driven Approach for Injury Type and Severity Detection with Hospital Recommendations for Emergency Response', justify=LEFT)
title.config(bg='pale turquoise', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Input Data Analysis", command=loadDataset)
uploadButton.place(x=1250,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Visualization", command=SampleDisplay)
processButton.place(x=1250,y=150)
processButton.config(font=font1) 

splitButton = Button(main, text="Data Splitting", command=splitDataset)
splitButton.place(x=1250,y=200)
splitButton.config(font=font1) 

MLButton = Button(main, text="Build & Train ML Models", command=MLModels)
MLButton.place(x=1250,y=250)
MLButton.config(font=font1)

EnsembleButton = Button(main, text="Build & Train Ensemble Model", command=EnsembleModel)
EnsembleButton.place(x=1250,y=300)
EnsembleButton.config(font=font1)

ProposedButton = Button(main, text="Build & Train Proposed Model", command=Proposed)
ProposedButton.place(x=1250,y=350)
ProposedButton.config(font=font1)

PerformanceButton = Button(main, text="Performance Graph", command=PerformanceGraph)
PerformanceButton.place(x=1250,y=400)
PerformanceButton.config(font=font1)

predictButton = Button(main, text="Injury Type & Severity Prediction", command=predict)
predictButton.place(x=1250,y=450)
predictButton.config(font=font1)

closeButton = Button(main, text="Close the Application", command=Exit)
closeButton.place(x=1250,y=500)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=45,y=100)
text.config(font=font1)

main.config(bg='lavender blush')
main.mainloop()
