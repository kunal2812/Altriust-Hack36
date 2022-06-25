import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from datetime import date
import os
import dlib
import cv2 as cv
import tempfile
from skimage import io
import matplotlib.pyplot as plt
import csv
from csv import writer
from csv import reader
import math
from PIL import Image, ImageOps
from skimage import io
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def img_to_encoding(image_path, model):
    img1 = cv.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def predict_score(img_path, EmoNet, show=False):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    model_class = np.argmax(EmoNet.predict(img_tensor), axis=-1)
    if model_class == 0:
      return int(3)
    elif model_class == 1:
      return int(2)
    else:
      return int(1)

def identify(image_path, database, model):

  encoding = img_to_encoding(image_path, model)
  min_dist = 100
  for (name, db_enc) in database.items():
    dist = np.linalg.norm(encoding-db_enc)

  if dist < min_dist:
    min_dist = dist
    identity = name
  if min_dist > 0.7:
    return "Not Found"
  return identity

def getCount():
  with open('./data/data.csv','r',newline="") as csvfile:
    writer_object = writer(csvfile)
    df=reader(csvfile)
    dg=pd.read_csv('./data/data.csv')
    cnt=-1
    flag=False
    name_list=[]
    count=0
    for col in dg:
      count=count+1
    return count, col

def add_to_database(NAME,DATE,VALUE):
  if not os.path.exists('./data/data.csv'):
    with open('./data/data.csv','w',newline='') as csvfile:
      data={'Name':[NAME],DATE:[VALUE]}
      df=pd.DataFrame(data,columns=['Name',DATE])
      df.to_csv('./data/data.csv',index=False,header=True)
  else:
    with open('./data/data.csv','r',newline="")  as csvfile:
      dg=pd.read_csv('./data/data.csv')
      df=reader(csvfile)
      flag1=False
      count1=-1
      y=0
      prevy=0
      for col in dg:
        count1=count1+1
        if col==DATE:
          flag1=True
          break
      count=-2
      flag=False
      for row in df:
        count=count+1
        if row[0]==NAME:
          flag=True
          y=row[count1]
          prevy=row[count1-1]
          break
      if flag==True:
        z=0
        if flag1==True:
          z=0

          if count1>=2:
            #z=max(int(VALUE),int(int(float(y))-int(float(prevy))))
            if y.len()==0:
              z=int(VALUE)
              z=int(z)+int(float(prevy))
            else:
              z=max(int(VALUE),int(int(float(y))-int(float(prevy))))
              z=int(z)+int(float(prevy)) 
          else:
            z=max(int(VALUE),int(float(y)))

          dg.loc[count,DATE]=int(z)
          dg.to_csv('./data/data.csv',index=False)
        else:
          dg[DATE]=""
          z=int(VALUE)
          if count1>=1:
            z=int(z)+int(float(y))
          dg.loc[count,DATE]=int(z)
          dg.to_csv('./data/data.csv',index=False)  
      else:
        with open('./data/data.csv','a+',newline="") as csvfile1:
          z=0
          if flag1==True:
            z=int(VALUE)
            if count1>=1:
              z=int(z)+int(float(prevy))
            dg.loc[count+1,'Name']=NAME
            dg.loc[count+1,DATE]=int(VALUE)
            dg.to_csv('./data/data.csv',index=False)
          else:
            dg[DATE]=" "
            z=int(VALUE)
            dg.loc[count+1,'Name']=NAME
            dg=pd.read_csv('./data/data.csv')
            dg.loc[count+1,DATE]=int(z)+int(float(prevy))
            dg.to_csv('./data/data.csv',index=False)

def scan_database(DATE, threshold):
  with open('./data/data.csv','r',newline="") as csvfile:
    writer_object = writer(csvfile)
    df=reader(csvfile)
    dg=pd.read_csv('./data/data.csv')
    cnt=-1
    flag=False
    name_list={}
    for col in dg:
      cnt=cnt+1
      if col==DATE:#date given by user
        break
    for row in df:
      if row[0]=="Name":
        continue
      if(int(float(row[cnt]))<threshold):
        name_list[row[0]] = int(float(row[cnt]))

  return name_list

def curr_value(DATE,NAME):
  with open('./data/data.csv','r',newline="") as csvfile:
    dg=pd.read_csv('./data/data.csv')
    df=reader(csvfile)
    count=-1
    flag=False;
    for col in dg:
      count=count+1
      if col==DATE:#date given by user
           break
    x=0
    for row in df:
      if row[0]==NAME:
         x=row[count]
       #print(x)
    return x

def view_database(database):
  today = date.today()
  d1 = today.strftime("%d/%m/%Y")
  for keys in database:
    st.subheader(keys)
    # value = 24
    c, d = getCount()    
    value = curr_value(d1,keys)
    value = math.ceil(((int(float(value))*1.0)/(3*c)) * 100)
    st.write(str(value) + "%")
    progress_bar = st.progress(value)
    if value < 40:
      status = st.write("Sad")
    else:
      status = st.write("Happy")

def scan_vdo(f, date):

  tfile = tempfile.NamedTemporaryFile(delete=False) 
  tfile.write(f.read())


  vf = cv.VideoCapture(tfile.name)

  stframe = st.empty()

  while vf.isOpened():
    ret, frame = vf.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imwrite("frame.jpg", frame)
    img_path = './frame.jpg'
    person = identify(tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48)), database, FrNet)
    if person!="Not Found":
      add_to_database(person,f_date,predict_score(img_path, EmoNet))

def save_uploaded_file(uploadedfile):

  img_path = os.path.join("./temps",uploadedfile.name)
  with open(os.path.join("./temps",uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return img_path

def main():

  def triplet_loss(y_true, y_pred, alpha = 0.2):    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    basic_loss = pos_dist- neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

  FrNet = tf.keras.models.load_model('./models/FrNet', custom_objects={'triplet_loss': triplet_loss})
  EmoNet = tf.keras.models.load_model('./models/emonet2.h5')
  database = {}
  phn_db = {}

  def initialize_db():

    today = date.today()
    # d1 = today.strftime("%d/%m/%Y")
    d1 = "11/4/2021"
    database["danielle"] = img_to_encoding("./images/danielle.png", FrNet)
    add_to_database("danielle", d1, predict_score("./images/danielle.png", EmoNet))
    phn_db["danielle"] = "8976442222"
    
    database["younes"] = img_to_encoding("./images/younes.jpg", FrNet)
    add_to_database("younes", d1, predict_score("./images/younes.jpg", EmoNet))
    phn_db["younes"] = "4354563465"

    database["tian"] = img_to_encoding("./images/tian.jpg", FrNet)
    add_to_database('tian', d1, predict_score('./images/tian.jpg', EmoNet))
    phn_db["tian"] = "25435345324"

    database["andrew"] = img_to_encoding("./images/andrew.jpg", FrNet)
    add_to_database("andrew", d1, predict_score("./images/andrew.jpg", EmoNet))
    phn_db["andrew"] = "3642646625"

    database["kian"] = img_to_encoding("./images/kian.jpg", FrNet)
    add_to_database("kian", d1, predict_score("./images/kian.jpg", EmoNet))
    phn_db["kian"] = "626434625"

    database["dan"] = img_to_encoding("./images/dan.jpg", FrNet)
    add_to_database("dan", d1, predict_score("./images/dan.jpg", EmoNet))
    phn_db["dan"] = "2636323443"

    database["sebastiano"] = img_to_encoding("./images/sebastiano.jpg",FrNet)
    add_to_database("sebastiano", d1, predict_score("./images/sebastiano.jpg", EmoNet))
    phn_db["sebastiano"] = "7364512323"

    database["bertrand"] = img_to_encoding("./images/bertrand.jpg", FrNet)
    add_to_database("bertrand", d1, predict_score("./images/bertrand.jpg", EmoNet))
    phn_db["bertrand"] = "342652655"

    database["kevin"] = img_to_encoding("./images/kevin.jpg", FrNet)
    add_to_database("kevin", d1, predict_score('./images/kevin.jpg', EmoNet))
    phn_db["kevin"] = "345423534"

    database["felix"] = img_to_encoding("./images/felix.jpg", FrNet)
    add_to_database("felix", d1, predict_score("./images/felix.jpg", EmoNet))
    phn_db["felix"] = "345134535"

    database["benoit"] = img_to_encoding("./images/benoit.jpg", FrNet)
    add_to_database("benoit", d1, predict_score("./images/benoit.jpg", EmoNet))
    phn_db["benoit"] = "21416778"

    database["arnaud"] = img_to_encoding("./images/arnaud.jpg", FrNet)
    add_to_database('arnaud', d1, predict_score("./images/arnaud.jpg", EmoNet))
    phn_db["arnaud"] = "0988987665"

  initialize_db()
  st.title("Altruist ~ We care <3")
  menu = ["Home", "Register into Database", "View Database", "Upload Video", "Scan Database"]
  choice = st.sidebar.selectbox('Menu', menu)
  if choice == 'Home':
    st.title("Home")
  elif choice == "Register into Database":
    st.title("Register into Database")
    name = st.text_input("Your full name")
    phn = st.text_input("Phone number")
    st.write("Upload a picture")
    img = st.file_uploader("Upload a picture",  type=["jpg", "png"])
    st.write('You selected `%s`' % img)
    if img is not None:
      file_details = {"FileName":img.name,"FileType":img.type}
      img_path = save_uploaded_file(img)
      st.write(img.name)
      submit = st.button("Submit")
      if submit:
        st.image(img, width=None)
        phn_db[name] = str(phn)
        img2 = tf.keras.preprocessing.image.load_img(img_path, target_size=(96, 96))
        img2 = img2.save("gg.jpg")
        img_path2 = "./gg.jpg"
        database[name] = img_to_encoding(img_path2, FrNet)

  elif choice == "View Database":
    st.title("View Database")
    view_database(database)
  elif choice == "Upload Video":
    st.write("Enter date on which footage was taken")
    f_date = st.text_input("DD/MM/YY")
    vdo = st.file_uploader("Upload a Video")
    st.write('You selected `%s`' % vdo)
    if vdo is not None:
      v = vdo.read()
      st.video(v)
      video_file = open(vdo, 'rb')
      video_bytes = video_file.read()
      st.video(video_bytes)
      scan_vdo(vdo, f_date)
  elif choice == "Scan Database":
    c, d = getCount()
    t = math.ceil(0.4*3*c)
    mylist = scan_database(d, t)
    view_database(mylist)     
    submit = st.button('Click to send text (*Frontend)')
    if submit:
      message()

if __name__ == '__main__':
  main()
