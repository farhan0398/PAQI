from tkinter import *
from tkinter import messagebox
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as mt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def bye():
       messagebox.showinfo("Alert","Have a nice day!")
       top.destroy()



def Random_Forest(met_train,met_test,aqi_train,aqi_test):
        model = RandomForestClassifier(n_estimators=266,criterion='gini',random_state=0)
        model.fit(met_train,aqi_train)
        accuracy = mt.accuracy_score(aqi_test, model.predict(met_test))*100
        return model, accuracy

       
def KNN(met_train,met_test,aqi_train,aqi_test):
        model = KNeighborsClassifier(n_neighbors=10,algorithm='auto')
        model.fit(met_train,aqi_train)
        accuracy = mt.accuracy_score(aqi_test,model.predict(met_test))*100
        return model, accuracy


def DT(met_train,met_test,aqi_train,aqi_test):
        model = DecisionTreeClassifier(criterion='entropy', random_state=0, max_leaf_nodes=6)
        model.fit(met_train,aqi_train)
        accuracy = mt.accuracy_score(aqi_test,model.predict(met_test))*100
        return model, accuracy

       

def get_data():
    browse = webdriver.Firefox(executable_path='C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\geckodriver.exe')
    browse.get("https://www.worldweatheronline.com/lang/en-in/new-delhi-weather/delhi/in.aspx")
    soup = BeautifulSoup(browse.page_source, "html.parser")

    temp = soup.find("div", {"class" : "carousel-cell well text-center"})
    temp = temp.findAll("address")
    temp = temp[0].text
    TM = temp[:2]
    Tm = temp[4:6]




    temp = soup.find("div", {"class": "tb_row tb_temp"})
    temp1 = temp.findAll("div", {"class": "tb_cont_item"})
    i = 0
    total = 0
    for value in temp1:
        if i > 0:
            data = value.text
            total += int(data[:2])
        i += 1
    T = total/4

    temp = soup.find("div",{"class":"col-lg-6 col-md-6 col-sm-6 col-xs-6 text-left"})
    text = temp.text
    text = text.split(" ")
    pp = text[1]
    h = text[3]
    h = h.split("%")
    h = h[0]
    slp = text[4]
    vv = text[7]

    temp = soup.find("div", {"class": "tb_row tb_wind"})
    temp1 = temp.findAll("div", {"class": "tb_cont_item"})
    i = 0
    vm = 0
    total = 0
    for value in temp1:
        if i > 0:
            data = value.text
            total += int(data[:2])
            if vm < int(data[:2]):
                vm = int(data[:2])
        i += 1
    v = total/4
    test = [T,TM,Tm,slp,h,pp,vv,v,vm]
    browse.close()
    met_train =  pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\met_data\\train.csv", usecols=[
        'T', 'TM', 'Tm', 'SLP', 'H','PP', 'VV', 'V', 'VM'])
    met_train.head()
    met_test = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\met_data\\test.csv",usecols=[
        'T', 'TM', 'Tm', 'SLP', 'H', 'PP', 'VV', 'V', 'VM'])
    aqi_train = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\pollutant_data\\train.csv",usecols=['AQI_Category'])
    aqi_test = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\pollutant_data\\test.csv",usecols=['AQI_Category'])
    aqi_test = aqi_test.values
    aqi_train = aqi_train.values
    return test,met_train,met_test,aqi_train,aqi_test




def KNNG():
       model, accuracy = KNN(met_train,met_test,aqi_train,aqi_test)
       plt.plot(np.array(np.arange(1,174,1)),model.predict(met_test),color='red')
       plt.plot(np.array(np.arange(1,174,1)),aqi_test,alpha=0.5)
       plt.title("actual value vs predicted value using KNN Classifier")
       plt.legend(('predicted value', 'actual value'),loc='lower left')
       plt.xlabel("days-->")
       plt.ylabel("Aqi category -->")
       plt.show()
       plt.show()


def RFCG():
       model, accuracy = Random_Forest(met_train,met_test,aqi_train,aqi_test)
       plt.plot(np.array(np.arange(1,174,1)),model.predict(met_test),color='red')
       plt.plot(np.array(np.arange(1,174,1)),aqi_test,alpha=0.5)
       plt.title("actual value vs predicted value using Random Forest Classifier")
       plt.legend(('predicted value', 'actual value'),loc='lower left')
       plt.xlabel("days-->")
       plt.ylabel("Aqi category -->")
       plt.show()
       plt.show()



def DTG():
       model, accuracy = DT(met_train,met_test,aqi_train,aqi_test)
       plt.plot(np.array(np.arange(1,174,1)),model.predict(met_test),color='red')
       plt.plot(np.array(np.arange(1,174,1)),aqi_test,alpha=0.5)
       plt.title("actual value vs predicted value using Decision Tree Classifier")
       plt.legend(('predicted value', 'actual value'),loc='lower left')
       plt.xlabel("days-->")
       plt.ylabel("Aqi category -->")
       plt.show()
       plt.show()



def RFC():
        model, accuracy = Random_Forest(met_train,met_test,aqi_train,aqi_test)
        result1 = model.predict(test)
        result2 = accuracy
        RC = Tk()
        RC.title("Random Forest Classifier")
        RC.geometry("900x400")
        
        r1=Label(RC,text="Random Forest Classifer\n",font=("Courier", 30))
        r1.pack()
        
        r2 = Label(RC, text="Category:", width=17, anchor='w',font=("Courier",15))
        r2.grid(row=0)
        r3 = Label(RC, text=result1)
        r3.grid(row=0, column=1)
        r4 = Label(RC, text="Accuracy:", width=17, anchor='w',font=("Courier",15))
        r4.grid(row=2)
        r5 = Label(RC, text=result2)
        r5.grid(row=2, column=1)
        Button(per, text='Graph', command= RFCG,font=("Courier")).place(rely=.3, relx=.5, anchor="center")
        RC.mainloop()
    


def KN():
        model, accuracy = KNN(met_train,met_test,aqi_train,aqi_test)
        result1 = model.predict(test)
        result2 = accuracy
        RC = Tk()
        RC.title("KNN Classifier")
        RC.geometry("900x400")
        
        r1=Label(RC,text="KNN Classifer\n",font=("Courier", 30))
        r1.pack()
        
        r2 = Label(RC, text="Category:", width=17, anchor='w',font=("Courier",15))
        r2.grid(row=0)
        r3 = Label(RC, text=result1)
        r3.grid(row=0, column=1)
        r4 = Label(RC, text="Accuracy:", width=17, anchor='w',font=("Courier",15))
        r4.grid(row=2)
        r5 = Label(RC, text=result2)
        r5.grid(row=2, column=1)
        Button(per, text='Graph', command= KNNG,font=("Courier")).place(rely=.3, relx=.5, anchor="center")
        RC.mainloop()



def Dec():
        model, accuracy = DT(met_train,met_test,aqi_train,aqi_test)
        result1 = model.predict(test)
        result2 = accuracy
        RC = Tk()
        RC.title("Decision Tree")
        RC.geometry("900x400")
        
        r1=Label(RC,text="Decision Tree\n",font=("Courier", 30))
        r1.pack()
        
        r2 = Label(RC, text="Category:", width=17, anchor='w',font=("Courier",15))
        r2.grid(row=0)
        r3 = Label(RC, text=result1)
        r3.grid(row=0, column=1)
        r4 = Label(RC, text="Accuracy:", width=17, anchor='w',font=("Courier",15))
        r4.grid(row=2)
        r5 = Label(RC, text=result2)
        r5.grid(row=2, column=1)
        Button(per, text='Graph', command= DTG,font=("Courier")).place(rely=.3, relx=.5, anchor="center")
        RC.mainloop()



        
       
def Predict():
       test_data = get_data()
       test = []
       test.append(test_data)
       met_train =  pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\met_data\\train.csv", usecols=[
        'T', 'TM', 'Tm', 'SLP', 'H','PP', 'VV', 'V', 'VM'])
       met_train.head()
       met_test = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\met_data\\test.csv",usecols=[
        'T', 'TM', 'Tm', 'SLP', 'H', 'PP', 'VV', 'V', 'VM'])
       print(met_test.head())
       aqi_train = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\pollutant_data\\train.csv",usecols=['AQI_Category'])
       aqi_test = pd.read_csv("C:\\Users\\adi08\\OneDrive\\Desktop\\CSE914909\\Project Code\\Data\\pollutant_data\\test.csv",usecols=['AQI_Category'])
       aqi_test = aqi_test.values
       aqi_train = aqi_train.values

       per = Tk()
       per.title("Passenger Details")
       per.geometry("400x300")
       p1=Label(per,text="Select The Classifier\n",font=("Courier", 20))
       p1.pack()
       Button(per, text='Random Forest Classifier', command= RFC,font=("Courier")).place(rely=.3, relx=.5, anchor="center")
       Button(per, text='KNN Classifier', command= KN,font=("Courier")).place(rely=.4, relx=.5, anchor="center")
       Button(per, text='Decision Tree', command= Dec,font=("Courier")).place(rely=.5, relx=.5, anchor="center")
       #B2 = Button(text = "Random Forest Classifier", command = RFC,font=("Courier", 20))
       #B1.place(rely=.63, relx=.5, anchor="center")
       #B2.config( height = 1 )
       per.mainloop()
       #return met_train,met_test,aqi_train,aqi_test

#--------------------------------------------------MAIN WINDOW----------------------------------------
top = Tk()
top.title("Prediction of Air Quality Index")
top.geometry("900x400")

t1=Label(top,text="Prediction of Air Quality Index\n",font=("Courier", 30))
t1.pack()
t=Label(top,text="MADE BY : Farhan Ansari\nCOLLEGE : Rizvi College of Arts Science and Commerce\n_______________________________________________________________________\nMAIN MENU",font=("Courier", 20))
t.pack()


#photo = PhotoImage(file="C:\\Users\\ASUS\\Pictures\\railways.png")
#w = Label(top, image=photo,compound='bottom',justify='center')
#w.pack()


B1 = Button(text = "Prediction", command = Predict,font=("Courier", 20))
B1.place(rely=.63, relx=.5, anchor="center")
B1.config( height = 1 )
B4 = Button(text = "EXIT", command = bye,font=("Courier", 20))
B4.place(rely=.8, relx=.5, anchor="center")
B4.config( height = 1 )


top.mainloop()
