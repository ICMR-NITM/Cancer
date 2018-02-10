
# coding: utf-8

# In[111]:


from Tkinter import *
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split




data1 = pd.read_csv('data.csv')


data1 = data1.drop(['Unnamed: 32','id'], axis=1)
y1 = data1.diagnosis 
x1 = data1.drop(['diagnosis'], axis=1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.0, random_state=42)

data = pd.read_csv('test_data.csv')
y = ['M']
x = data
X_test, X_train, y_test, y_train = train_test_split(x, y, test_size=0.0, random_state=42)


svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(x1,y1)
y_pred = svc.fit(X_train1, y_train1).predict(X_test)
out = pd.DataFrame(svc.predict_proba(X_test))
out.head()



def for_user():
    
    color = ""
    message = ""
    p_benign = 0
    p_malignant = 0

    is_safe = ""
    img = ""

    i=0
    count=0
    for index, row in out.iterrows():
        print row[0], row[1]

        if row[0]<row[1]:
            message = "The predicted tumour is cancerous !!"
            color = "red"
            p_benign = round(row[0]*100, 2)
            p_malignant = round(row[1]*100, 2)
            is_safe = "Patient is not safe."
            img = "images/sad.gif"

        elif row[0]>row[1]:
            message = "The predicted tumour is non-cancerous !!"
            color = "green"
            p_benign = round(row[0]*100, 2)
            p_malignant = round(row[1]*100, 2)
            is_safe = "Patient is safe."
            img = "images/smile.gif"




    root = Tk()
    root.configure(background='black')
    root.geometry('{}x{}'.format(800, 500))
    root.title('Cancer prediction using machine learning')

    label = Label(root, text=message, bg="black", fg=color, font=(None, 25))
    label.pack()

    prob_b ="Probablity of being Benign is : " + str(p_benign)[:6] +"%"
    label_b = Label(root, text=prob_b, bg="black",fg=color, font=(None, 25))
    label_b.pack()

    prob_m ="Probablity of being Malignant is : "+ str(p_malignant)[:6]+"%"

    label_m = Label(root, text=prob_m, bg="black",fg=color, font=(None, 25))
    label_m.pack()

    t = is_safe
    

    photo = PhotoImage(file=img)
    label = Label(image=photo)
    label.image = photo 
    label.pack()

    label_t = Label(root, text=t, bg="white",fg=color, font=(None, 25))
    label_t.pack()

    root.mainloop()
    
for_user()

