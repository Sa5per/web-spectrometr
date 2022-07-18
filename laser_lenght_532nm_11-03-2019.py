###imports
import cv2
import pickle
import os
import random
#import math
import time
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import messagebox
from tkinter import filedialog as fd
import pandas as pd
import threading

import matplotlib
#matplotlib.use('TkAgg')

import tkinter.ttk as ttk
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#import serial.tools.list_ports
#--------------------------------------------------------

###GUI
window = tk.Tk()  #Makes main window
window.wm_title("Анализатор длины волны v.5 (19-03-2019)")
window.config(background="#FFFFFF")

imageFrame = tk.Frame(window, width=1280, height=720)
imageFrame.grid(row=0, column=0, padx=5, pady=5)

imageFrame = Label(imageFrame)
imageFrame.grid(row=20, column=20)

#--------------------------------------------------------
config=[] #настройки из файла
typeBlur=0 #0-GaussianBlur,1-medianBlur
kernelBlur=15 #15- for GaussianBlur,5 for medianBlur               

redline=5 #длина красной линии в nm в природе
redline_def_val=20
redline_max=254 #950nm - максимальная регистрируемая длина волны (предположим)
greenline_def_val=614
greenline=532 #длина зеленой линии в природе
blueline_max=350

#загрузка рандомной картинки из каталога
folder=[]
last_dir=[]
last_file=[]
pic_file=''
file_repeat=0 #показываем однажды показанную?
dir_repeat=0 #выбираем тот же самый каталог?
pic_last_time=0
pic_time_val=10
pic_start_val=0
pic_list=[] # список всех показанных картинок
pic_file_list=[] #список показанных картинок с полными путями

def init_folder(): #получаем дерево каталогов
    global folder
    
    for i in os.walk(".\images"):
       folder.append(i)
    return()

def get_random_file():
    global folder
    global last_dir
    global last_file
    global file_repeat
    global dir_repeat
    global pic_file
    global pic_file_list
    global pic_list
    global dir_last
    
    directory= random.choice(folder[1:-1])
    if dir_repeat==0:
        if len(last_dir)>0:
            while last_dir[-1]==directory[0]:
                directory= random.choice(folder[1:-1])    
    last_dir.append(directory[0])

    file= random.choice(directory[2])
    if file_repeat==0: #повторы включены?
        if len(last_file)>0:
            i=0
            while (file in last_file):  # если такую картинку уже показывали, то               
                if i>len(directory[2]): #смотрим счетчик, если  больше, чем картинок в категории (каталоге), то
                    #print(directory[2])
                    for j in directory[2]: 
                        #print('remove: ',j)
                        if j in last_file:
                            last_file.remove(j) # удаляем все файлы из данной категории из списка last_file 
                file= random.choice(directory[2])
                i=i+1
                    
    last_file.append(file) #рабочий список,из которого удаляются дубликаты и прочее (т.е. нельзя использовать для итоговогосписка показанных картинок)
    pic_list.append(file) #список показанных картинок без дубликатов
    pic_file=directory[0]+'\\'+file
    pic_file_list.append(pic_file) #список файлов с полным путем
    
    return()

def show_random_pic():

    get_random_file()
    img = cv2.imread(pic_file,1)
    cv2.imshow('image',img)
    print(pic_file)
    return()

def pic_view():
    if pic_start.get():
      img=cv2.imread('.\images\start_fon.jpeg',1)
      cv2.imshow('image',img)

#загружаем конфиг файл
def offconfig():
    global config
    global typeBlur
    global kernelBlur
    global cam_name
    global redline_def_val
    
    configArray=[]
    

    file_name='config.ini'
    try:
            file = open(file_name,'r').read()
    except IOError as e:
            print(u'файл с конфигом не существует')
    else:
            print(u'загружаем конфиг')
            configArray = file.split('\n')
            for eachLine in configArray:
              if len(eachLine)>0:
                config.append(int(eachLine))
                print(config[-1])
            typeBlur=config[0]
            kernelBlur=config[1]
            redline_def_val=config[2]
            cam_name=config[3]
            if typeBlur==1:
                print('medianBlur \n')
            elif typeBlur==0:
                print('GaussianBlur n')
            print('kernel=',kernelBlur)
            if cam_name==90:
               print('camera: 192.168.0.90')

#spectr_lenght=600  #длина спектра в px. по расчетам 32px=1nm, значит диапазон измерений длины волны= 600/32=18,75nm примерно
spectr_lenght=600  #длина спектра в px. по расчетам 12.2577px=1nm, значит диапазон измерений длины волны= 600/12.257=49nm примерно

spec_graph= np.zeros((300,spectr_lenght,3), np.uint8)

### толщина накопления спектрального значения
thinSpec=300

### кол-во кадров для усреднения показаний
timespec=24

###CAM properties
cam_width  = 1280
cam_height = 720
cam_fps = 30
resize_pic=False #следует уменьшать картинку? да или нет
cam_name=0
offconfig()
frame=[] #буер для кадра из камеры
camera_stop=False #переменная для выключения процесса получения кадров


#--------------------------------------------------------

Overlay1_color = (200, 200, 200)
Overlay2_color = (220, 220, 220)

#Массив для гистограммы
hist=[]
#Спектрограмма и буфер для записи времени каждой спектрограммы
spec_array=[]
timepoints=[]
#начало и конец спектра в nm
spec_start=blueline_max
spec_end=950

#включение спектра? 1- вкл, 0-выкл
save_spectr=0
only_spectr_on=0

#файл для сохранения картинки
file_name='spectr.png'
#флаг записи кадра изображения
savefig=0

#флаг метки (если нажата кнопка "добавить метку",то =1):
metka=False
#массив меток (указывается номер спектрального среза)
metkabuf=[]
#массив временнЫх отметок времени каждой метки
metkatime=[]

#создаем переменную для отсчета времени
start_time= time.time()

# параметры цветового фильтра
hsv_yellow = np.array((10, 0, 0), np.uint8) #оранжевый цвет
hsv_blue = np.array((110, 255, 255), np.uint8)# голубо-синий

#массив рассчитанной длины волны
filter_data=0
ydata=[]

#массив цветов для отметок
rgb=['red','green','orange','magenta','yellow','black','red','green','blue','red','green','blue','red','green','blue','red','green','blue']

               
def spectrOnOFF():
    global save_spectr
    global start_time
    global metkabuf
    global hist
    global timepoints
    global pic_last_time
    global pic_time_val
    global pic_start_val
    global last_file
    global last_dir
    global pic_list
    
    if save_spectr==0:
        save_spectr=1
        spectrButton.configure(text='Откл.запись')
        #очищаем буферы спектра и времени
        hist.clear()
        timepoints.clear()
        metkabuf.clear()
        metkatime.clear()
        ydata.clear()
        last_file.clear()
        last_dir.clear()
        pic_list.clear()
        
        start_time=time.time();
        pic_last_time=start_time
        pic_time_val=int(pic_time.get())
       # print(pic_time_val)
        pic_start_val=pic_start.get()
       # print(pic_start_val)
        if pic_start_val:
            show_random_pic()
        
        
    else:
        save_spectr=0
        pic_start.set(0)
        print(pic_list)
        spectrButton.configure(text='Вкл.запись')
    
def video_grab():
    global frame
    print("video thread start..")
    while (camera_stop==False):
        camera.grab()
        _,frame = camera.retrieve()
    print("video thread stoped")
        
    
def VideoCapture():
    global cropping_width_def_val
    global cropping_height_def_val
    global hist
    global spec_array
    global save_spectr
    global only_spectr_on
    global savefig
    global start_time
    global metka
    global metkabuf
    global metkatime
    global timepoints
    global ydata
    global filter_data
    global kernelBlur
    global typeBlur
    global resize_pic
    global frame
    global pic_last_time
    global pic_time_val
    global pic_start_val
    global pic_list
    
    
    #Get the parameters value for cropping from GUI
    redline_val=redlinePos.get()
    greenline_val=greenlinePos.get()
    specline_val=speclinePos.get()
    anglespec_val=angleSpec.get()
    
    
    if(camera.isOpened()):
        #Read video stream from camera происходит в фоновом режиме

        M = cv2.getRotationMatrix2D((cam_width/2,cam_height/2),anglespec_val,1)

        #Если размер кадра оказался больше 1280х720, то уменьшаем картинку
        if (resize_pic==True)&(frame.shape[0]>720):
            frame=frame[180:900,320:1600]
        frameAngle = cv2.warpAffine(frame,M,(cam_width,cam_height))

        #Use frame in HSV mode, для выборки интенсивностей        
        hsv = cv2.cvtColor(frameAngle, cv2.COLOR_BGR2HSV )

        #высчитываем реальную длину спектра на картинке с камеры        
        startspectr=greenline_val-300
        endspectr=greenline_val+300
        
        #делаем ч/б картинку для посика границ
        grayscaled = cv2.cvtColor(frameAngle,cv2.COLOR_BGR2GRAY)
        if typeBlur==0:
            blur = cv2.GaussianBlur(grayscaled,(kernelBlur,kernelBlur),0)
        elif typeBlur==1:
            blur = cv2.medianBlur(grayscaled,kernelBlur,0)
        retval, thresh = cv2.threshold(blur, redline_val,255 , cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        
        # ищем границы светового пятна:
        _, contours,hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        new_contours=[]
        for c in contours:
          if cv2.contourArea(c)>500:
             new_contours.append(c)
        if only_spectr_on==0:
            cv2.drawContours( frameAngle, new_contours, -1, (255,0,0), 3 )

        best_box=[-1,-1,-1,-1]
        for c in new_contours:
            x,y,w,h = cv2.boundingRect(c)
            if x>startspectr and x<endspectr and w<int((endspectr-startspectr)/2):
                if best_box[0] < 0:
                    best_box=[x,y,x+w,y+h]
                else:
                    if x<best_box[0]:
                        best_box[0]=x
                    if y<best_box[1]:
                        best_box[1]=y
                    if x+w>best_box[2]:
                        best_box[2]=x+w
                    if y+h>best_box[3]:
                        best_box[3]=y+h
        middle_line=int((best_box[2]-best_box[0])/2+best_box[0])
        
        if only_spectr_on==0:
            cv2.rectangle(frameAngle,(best_box[0],best_box[1]),(best_box[2],best_box[3]),(0,0,255),2)
        #рассчитываем длину волны
        if middle_line>startspectr and middle_line<endspectr:
            if only_spectr_on==0:
                cv2.line(frameAngle,(middle_line,specline_val-30),(middle_line,specline_val+30),(255,255,255),2)
            if middle_line<greenline_val:
                filter_data=532-(greenline_val-middle_line)/32
            if middle_line>greenline_val:
                filter_data=532+(middle_line-greenline_val)/32
            
        else:
            filter_data=532.0

        #Display the new resolution to work with
        cv2.putText(frameAngle,"{}x{}".format(cam_width, cam_height),(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1, cv2.LINE_AA)
       
        #рисуем область подсчета    спектра
        if only_spectr_on==0:
            cv2.rectangle(frameAngle,(startspectr,specline_val-int(thinSpec/2)),(endspectr,specline_val+int(thinSpec/2)),(0,255,0),2)

        #вырезаем кусок со спектром
        #вырезаем кусок равный длине спектра.формат картинки(массива) [y,x]
        crop_img=hsv[specline_val-int(thinSpec/2):specline_val+int(thinSpec/2),startspectr:endspectr]

        #очищаем все буфера 
        spec_array.clear()
        spec_win = crop_img
        spec_graph.fill(0)

     #Создаем гистограмму яркости линий спектра в отдельный буфер
        for x in range(spec_win.shape[1]): # spec_win.shape[1] - значение Х. если просто подставить 1280, то будет ошибка
            i=0
            for y in range(spec_win.shape[0]):
               i=i+spec_win.item(y,x,2)
            spec_array.append(int(i/spec_win.shape[0]))
            spec_graph.itemset((268-spec_array[x],x,2),255)

       #если нажата кнопка записи, то пополняем спектр новыми данными
        if save_spectr==1:

            time_now=time.time()
            delta_time=time_now-start_time

            #считаем время, прошедшее с показа последней картинки
            delta_pic_time=time_now-pic_last_time
            
            #покажем случайную картинку
            if pic_start_val==True:
                if ((delta_pic_time)%3600%60)>pic_time_val:                    
                    show_random_pic()
                    pic_last_time = time_now
                    metka=True

            #добавляем гистограмму в буфер + добавляем время каждой гистограммы
            timepoints.append(delta_time)
            hist.append(spec_array[0:-1]) #именно так [0:-1], иначе ошибка при отрисовке графика по времени(рисует только последнюю строку)
            ydata.append(filter_data)
            #если добавили очередную отметку, то:           
            if metka==True:
                metkabuf.append(len(hist))
                metkatime.append(delta_time)
                #print(metkatime[-1])
                metka=False
            #выводим время на экран спектрографа
            h=(delta_time)/3600 
            m=(delta_time)%3600/60 
            s=(delta_time)%3600%60
            time_text='%i:%i:%i   metok: %i' % (h, m, s,len(metkatime))
            cv2.putText(spec_graph,time_text,(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
       
                
            
        #Display red,green lines
        if only_spectr_on==0:
          cv2.line(frameAngle, (redline_val, 0), (redline_val, cam_height), (0,0,254), 1, cv2.LINE_AA)
          cv2.line(frameAngle, (greenline_val, 0), (greenline_val, cam_height), (0,254,0), 1, cv2.LINE_AA)
          cv2.line(frameAngle, (0,specline_val), (cam_width,specline_val), (254,254,254), 2, cv2.LINE_AA)
          #Set back the frame to color to display on GUI
          color = cv2.cvtColor(frameAngle, cv2.COLOR_BGR2RGBA)

        else:
          cv2.line(spec_graph,(1,0),(1,290),(255,255,255), 1, cv2.LINE_AA)
          cv2.line(spec_graph,(0,280),(spectr_lenght,280),(255,255,255), 1, cv2.LINE_AA)
          for i in range(10,255,10):
              cv2.line(spec_graph,(0,280-i),(spectr_lenght,280-i),(50,50,50), 1, cv2.LINE_AA)
              #cv2.putText(spec_graph,str(i+5),(10,285-i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,250,250), 1, cv2.LINE_AA)
              
          #рисуем сетку (подписи под графиком)
          for i in range(300,spectr_lenght,32):
              cv2.line(spec_graph,(i,280),(i,285),(255,255,255), 1, cv2.LINE_AA)
              cv2.putText(spec_graph,str(int((i-300)/32)+532),(i,295), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (250,250,250), 1, cv2.LINE_AA) #12.257px=1нм. спектр выводим с 532нм
          for i in range(268,0,-32):
              cv2.line(spec_graph,(i,280),(i,285),(255,255,255), 1, cv2.LINE_AA)
              cv2.putText(spec_graph,str(532-int((300-i)/32)),(i,295), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (250,250,250), 1, cv2.LINE_AA) #12.257px=1нм. спектр выводим с 532нм

          color = cv2.cvtColor(spec_graph, cv2.COLOR_BGR2RGBA)
           
        #TkInter process for displaying the video
        img = Image.fromarray(color)
        imgtk = ImageTk.PhotoImage(image=img)
        display1.imgtk = imgtk
        display1.configure(image=imgtk)
        
        
    window.after(50, VideoCapture)

            
    
    
def onconfig():
    global only_spectr_on

    imagePreviewFrame.configure(width=cam_width, height=cam_height)
    CameraFrame.grid()
    SpecialFrame.grid()
    angleFrame.grid()
    HeadBoxFrame.grid()
    
    only_spectr_on=0

def only_spectr():
    
    global only_spectr_on

    imagePreviewFrame.configure(height=305,width=(600+5))
    CameraFrame.grid()
    SpecialFrame.grid_remove()
    angleFrame.grid_remove()
    HeadBoxFrame.grid_remove()
    
    only_spectr_on=1

def yformat(y):
    global timepoints

    #print(len(timepoints))
    if y<=0:
        return y
    elif y<=len(timepoints) and y>0:
        m=(timepoints[int(y)])%3600/60
        s=(timepoints[int(y)])%3600%60
        time_last='%i:%i' % (m, s)
        return time_last
    else:
        m=(timepoints[-1])%3600/60
        s=(timepoints[-1])%3600%60
        time_last='%i:%i' % (m, s)
        return time_last

def viewSpectr():
    global hist
    global metkabuf
    global timepoints
    global savefig
    global ydata
    time_ylabel=[]
    time_ylabel.clear()

    viewLaser()
 
    if len(hist)>0:
        #plt.figure(figsize=(4,2))
        fig=plt.figure()
        ax1 = fig.add_subplot(111)
        plt.imshow(hist,cmap='jet',interpolation='nearest',aspect='auto',origin='lower')
        #plt.xticks(np.arange(0,redline_max-blueline_max,100),np.arange(blueline_max, redline_max, step=100))
        plt.xticks([78.6,103.2,127.8,152.4,177,201.6,226.2,250.8,275.4,300,324.6,349.2,373.8,398.4,423,447.6,472.2,496.8,521.4],[523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541])

        yax=ax1.get_yticks()
        #print (yax)
        for y in yax:
        #    print (y)
            time_ylabel.append(yformat(y))
        ax1.set_yticklabels(time_ylabel)
        
        #ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
        #покажем время в минутах:секундах
        m=(timepoints[-1])%3600/60 
        s=(timepoints[-1])%3600%60
        time_last='%i:%i' % (m, s)
        
        #time_ylabel.append('0:0')

        plt.xlabel('nm')
        plt.ylabel('time[s]')
        
        plt.colorbar(orientation='horizontal')
        i=0
        if len(metkabuf)>0:
            i=0
            while i < len(metkabuf):
              plt.axhline(y=metkabuf[i],color='black')
              m=(timepoints[metkabuf[i]])%3600/60 
              s=(timepoints[metkabuf[i]])%3600%60
              time_text='%i:%i' % (m, s)
              plt.text(5,metkabuf[i]+10,time_text,bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}) #выводим над линией метки время самой метки)
              #time_ylabel.append(time_text)
        
              i=i+1
        #time_ylabel.append(time_last)
        #plt.text(
        if savefig==1:
           plt.savefig(file_name)
           savefig=0
        plt.show()
    


def viewLaser():    #График распознанной длины волны  
    global metkabuf
    global timepoints
    global savefig
    global ydata
    global file_name
    global last_file
    global pic_list
    
    print(last_file)
    if len(ydata)>0:#если данные приняты (длина буфера больше 0), то строим графики
        #print ('выводим второй график')
# первый график Принятого сигнала
        plt.figure(num=2,figsize=(15, 6),clear=True)
        plt.subplots_adjust(left=0.04,bottom=0.14,right=0.96,top=0.9)
                
        if len(metkabuf)<1:
            plt.plot(timepoints, ydata,ls=':')# вывод точек интерполяции,'o',interp_timepoints, interp_sig,'.')
            plt_text(0,-1,-2) #выводим на экран среднее значение и среднекв.откл
        else:
            plt.plot(timepoints[:metkabuf[0]],ydata[:metkabuf[0]],ls=':')
            plt_text(0,metkabuf[0],0)
        plt.title('Длина волны')
        plt.xlabel('Время [сек]')
        plt.ylabel('Длина волны[nm]')
        plt.grid(True)# включаем сетку

#2-i график - сигнал между отметками       
        if len(metkabuf)>0:
                i=1
                while i < len(metkabuf):

                    plt.plot(timepoints[metkabuf[i-1]:metkabuf[i]],ydata[metkabuf[i-1]:metkabuf[i]],color=np.random.rand(3),ls=':')#color=rgb[i-1],ls=':')
                    plt_text(metkabuf[i-1],metkabuf[i],i)#выводим на экран среднее значение
                    i=i+1
                    
          # выводим сигнал от последней метки до конца сигнала
                #plt.figure(1)
                plt.plot(timepoints[metkabuf[i-1]:],ydata[metkabuf[i-1]:],color=np.random.rand(3),ls=':')#color=rgb[i-1],ls=':')
                plt_text(metkabuf[i-1],-1,i)#выводим на экран среднее значение и название категории
              

#выводим график плавающего среднего по 100 отсчетам
        mean_data=pd.Series(ydata,index=timepoints)
        ma=mean_data.rolling(10).mean()

        plt.plot(ma.index,ma,color='black',label='Среднее')
        plt.legend()


        if savefig==1:
            plt.savefig(file_name)
        if savefig==0:
            plt.show()
        plt.close()

    
def plt_text(x1,x2,y):#выводим на экран среднее значение
    #global sig
    global timepoints
    global pic_list
    

    xm = np.mean(ydata[x1:x2])  # среднее значение
    xs = np.std(ydata[x1:x2])
    x_pos=int(timepoints[x1]+(timepoints[x2]-timepoints[x1])/2)
    # выводим среднее
    plt.text(x_pos, xm+xs*2, 'среднее: %.3f'%(xm), verticalalignment='bottom', horizontalalignment='left',color='black',bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    # выводим название картинки
    #print (y)
    maxy=plt.ylim()
    
    if len(pic_list)>0:
        #print(y)
        plt.text(x_pos, maxy[0], pic_list[y], verticalalignment='bottom', horizontalalignment='left',color='black',bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})


def saveSpecPng(file_name):
    plt.savefig(file_name)


def saveSpectr():

    global savefig
    global file_name
    global timepoints
    global metkabuf
    global ydata
    global spec_graph
    global pic_list
    global pic_file
    global pic_file_list

    #print (last_file)
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = fd.asksaveasfilename(initialfile=timestr,defaultextension='*.sfs',filetypes=(("SPECTR files", "*.sfs"),
                                                ("All files", "*.*") ))
#записываем файл с буфером времени каждого спектра
    if file_name : #записываем данные в файл
        with open(file_name,'wb') as f:
           pickle.dump(hist,f)
           
        file_name=file_name[:-3]+'csv'
#записываем данные длины волны в файл
        f = open(file_name, 'w')
        i=0
        while i < len(ydata):
            f.write(str(timepoints[i])+';'+str(ydata[i])+'\n')
            i +=1
        f.close()
        
#записываем метку+названия категорий
        file_name=file_name[:-3]+'kat'
        f = open(file_name, 'w')
        #print ("категории: ",last_file)
        i=0
        while i < len(pic_file_list):
            if i==0:
                f.write(str('0'+';'+pic_file_list[i]+'\n'))
            else:
                f.write(str(metkabuf[i-1])+';'+str(pic_file_list[i])+'\n')
            i +=1
        f.close()



        file_name=file_name[:-3]+'png'
        savefig=1
        viewLaser()
        savefig=0

    return

def loadSpectr():
    global ydata
    global timepoints
    global metkabuf
    global spec_graph
    global pic_list
    global pic_file_list
    
    file_name = fd.askopenfilename(defaultextension='*.csv', filetypes=(("CSV files", "*.csv"),
                                                ("All files", "*.*")))
    if file_name :
        f = open(file_name, 'r').read()
        timepoints=[]
        ydata=[]
        dataArray = f.split('\n')

    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(';')
            timepoints.append(float(x))
            ydata.append(float(y))
            #print("x="+x+" y="+y+"\n")
    #f.close()
    
    if file_name:

        #пробуем загрузить названия категорий
        metka_file=file_name[:-3]+'kat' 
        try:
            file = open(metka_file,'r').read()
        except IOError as e:
            print(u'файл с категориями не существует')
            metkabuf=[]
            pic_list=[]
            pic_file_list=[]
        else:
            #print(u'заносим метки в базу данных')
            buf=[]
            metkabuf=[]
            pic_list=[]
            pic_file_list=[]
            metkaArray = file.split('\n')
            for eachLine in metkaArray:
              if len(eachLine)>0:
                buf=eachLine.split(';') #делим полученную строку на 2 части. разделитель ";"
                metkabuf.append(int(buf[0]))#первая часть - отметка. заносим её в буфер

                #теперь получаем список файлов для каждой отметки
                pic_file_list.append(str(buf[-1]))
                pic_file_name=pic_file_list[-1].split('\\')#делим строку на части, где разделить "\"
                pic_list.append(pic_file_name[-1])# и записываем последню часть полученного масива (а это имя файла) в список показанных файлов

            del metkabuf[0]#удаляем первый элемент метки, т.к. он не используется

            
    cv2.rectangle(spec_graph,(0,0),(600,30),(0,0,0),-1)
    cv2.putText(spec_graph,'file loaded',(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
    color = cv2.cvtColor(spec_graph, cv2.COLOR_BGR2RGBA) #преобразуем массив в картинку
    img = Image.fromarray(color)
    imgtk = ImageTk.PhotoImage(image=img)
    display1.imgtk = imgtk
    display1.configure(image=imgtk)
    
def metkaFun():
    global metka
    metka=True   

### инициализация видео
if cam_name==0:
    camera = cv2.VideoCapture(cam_name)
elif cam_name==90:
    camera = cv2.VideoCapture("rtsp://192.168.0.90:554/user=admin&password=&channel=1&stream=0.sdp?")
#Give some time for the camera to start
time.sleep(1)

if (camera.isOpened()):
    print ("Camera opened")
    camera.grab()
    _,frame=camera.retrieve() #получим один кадр для начала
else:
    print("camera not found")

#Set camera resolution and frame/sec if possible
if camera.get(cv2.CAP_PROP_FRAME_WIDTH)>1280:
    resize_pic=True
elif camera.get(cv2.CAP_PROP_FRAME_WIDTH)<1280:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    camera.set(cv2.CAP_PROP_FPS, cam_fps)
    resize_pic=False
elif camera.get(cv2.CAP_PROP_FRAME_WIDTH)==1280:
    resize_pic=False

#   Запускаем грабер кадров в фоновом режиме
thread_video= threading.Thread(target=video_grab,name="video")
thread_video.start()

#запускаем инициализацию дерева каталогов
init_folder()

###Control Frame
Controls = LabelFrame(imageFrame, text='Управление')
Controls.grid(row=0, column=0, rowspan=10, ipadx=5, ipady=5, sticky='nesw')

###BlankSpace
blankspace = Label(Controls)
blankspace.grid(row=0, column=0, sticky='we')

###HitBoxFrame
angleFrame = LabelFrame(Controls, text='Угол поворота')
angleFrame.grid(row=1, column=0 ,sticky='nesw', ipadx=5, ipady=5, columnspan=1)
#Label(HitBoxFrame, text="Min Size").grid(row=0, column=1, sticky='ws')
angleSpec = Scale(angleFrame, from_=-10, to=10, resolution=0.1, orient=HORIZONTAL)
angleSpec.grid(row=0, column=0, sticky='w')
angleSpec.set(0.0)

###Положение спектра
SpecialFrame = LabelFrame(Controls, text='Линия спектра')
SpecialFrame.grid(row=3, column=0 ,sticky='nesw', ipadx=5, ipady=5)
Label(SpecialFrame, text="Y").grid(row=0, column=1, sticky='wn')
speclinePos = Scale(SpecialFrame, from_=100, to=600,length=cam_height-300)
speclinePos.grid(row=0, column=0, sticky='n')
speclinePos.set(300)


### Кнопка сокрытия элементов управления

offButton = Button(Controls, width=16, height=1, text='Загрузить настройки', command = offconfig)
offButton.grid(row=4, column=0, padx=2, pady=2, sticky='nesw')

onButton = Button(Controls, width=16, height=1, text='Показать настройки', command = onconfig)
onButton.grid(row=5, column=0, padx=2, pady=2, sticky='nesw')

onSpectr = Button(Controls, width=16, height=1, text='Скрыть настройки', fg="red", command = only_spectr)
onSpectr.grid(row=6, column=0, padx=2, pady=2, sticky='nesw')

#Автоматический показ кртинок

pic_start = BooleanVar()
pic_start.set(0)
pic = Checkbutton(Controls,text="Автопоказ картинок", variable=pic_start, onvalue=1, offvalue=0,command=pic_view)
pic.grid(row=7, column=0, padx=2, pady=2, sticky='nesw')

#Время показа картинки
label_time=Label(Controls,text='Сек. на кртинку: ')
label_time.grid(row=8,column=0,padx=2,sticky='w')

pic_time=Entry(Controls)
pic_time.grid(row=9,column=0, padx=2)
pic_time.insert(0, "10")


### Вкл/выкл запись спектра в память
spectrButton = Button(Controls, width=16, height=1, text='Вкл.запись', fg="red", command = spectrOnOFF)
spectrButton.grid(row=10, column=0, padx=2, pady=2, sticky='nesw')

### Добавить метку
metkaButton = Button(Controls, width=16, height=1, text='+Метка', command = metkaFun)
metkaButton.grid(row=11, column=0, padx=2, pady=2, sticky='nesw')

### Сравнить спектры
#metkaButton = Button(Controls, width=16, height=1, text='Сравнить спектры', command = spectrDiff)
#metkaButton.grid(row=9, column=0, padx=2, pady=2, sticky='nesw')

### Вывод графика спектра во ремени на экран
viewButton = Button(Controls, width=16, height=1, text='Вывод графика', command = viewLaser)
viewButton.grid(row=12, column=0, padx=2, pady=2, sticky='nesw')

### Запись данных в файл
saveButton = Button(Controls, width=16, height=1, text='Сохранить', command = saveSpectr)
saveButton.grid(row=13, column=0, padx=2, pady=20, sticky='s')

loadButton = Button(Controls, width=16, height=1, text='Загрузить', command = loadSpectr)
loadButton.grid(row=14, column=0, padx=2, pady=2, sticky='s')

### CameraFrame
CameraFrame = LabelFrame(imageFrame, width=cam_width, height=cam_height)
CameraFrame.grid(row=2, column=1, ipadx=5, ipady=5, columnspan=2)
imagePreviewFrame = tk.Frame(CameraFrame, width=cam_width, height=cam_height)
imagePreviewFrame.grid(padx=5, pady=5, sticky='nesw')
imagePreviewFrame.grid_propagate(0)
display1 = tk.Label(imagePreviewFrame)
display1.grid(row=0, column=0, sticky='nesw')
display1.place(in_=imagePreviewFrame, anchor="c", relx=.5, rely=.5)

###HeadBox Frame
HeadBoxFrame = LabelFrame(imageFrame, text='Положение линий')
HeadBoxFrame.grid(row=1, column=1,columnspan=2,sticky='nesw')
Label(HeadBoxFrame, text="Thresh",fg='red').grid(row=0, column=0, sticky='sw')

redlinePos = Scale(HeadBoxFrame, from_=1, to=254,length=cam_width-100,resolution=1, orient=HORIZONTAL)
redlinePos.grid(row=0, column=1,sticky='we')
redlinePos.set(redline_def_val)

Label(HeadBoxFrame, text="Green 532нм:",fg='green').grid(row=1, column=0, sticky='sw')

greenlinePos = Scale(HeadBoxFrame, from_=1, to=cam_width,length=cam_width-100,resolution=1, orient=HORIZONTAL)
greenlinePos.grid(row=1, column=1, sticky='we')
greenlinePos.set(greenline_def_val)



#When closing application
def on_closing():
    global camera_stop
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        camera_stop=True
        thread_video.join()
        camera.release()
        time.sleep(1)
        cv2.destroyAllWindows()
        window.destroy()

#Combo.bind('<<ComboboxSelected>>', on_select)
window.protocol("WM_DELETE_WINDOW", on_closing)
VideoCapture()      #spectr recognition
window.mainloop()   #Starts GUI
