from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import fbank
from scipy import signal as sg
import librosa
import scipy.io.wavfile as wav
from pyAudioAnalysis import audioFeatureExtraction
import os.path as path
import sounddevice as sd
import threading
from csv import reader
from multiprocessing import Process
import time
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf
import time
from pydub import AudioSegment
from pydub import AudioSegment, silence
import wave
import numpy as np
import warnings
import gc
from scipy import fftpack
import pylab as plt
import warnings
import datetime
import pandas as pd
import csv
from scipy.signal import fftconvolve
from scipy.signal import convolve
import pandas as pd
#import ray
#ray.init()

warnings.simplefilter("ignore", DeprecationWarning)

#############################################3

def show(data, data2):
    import numpy as np
    import wave
    import pylab as plt
    import random
    import struct
    warnings.simplefilter("ignore", DeprecationWarning)
    signal = np.frombuffer(data, 'Int16')
    signal2 = np.frombuffer(data2, 'Int16')
    plt.figure(1)
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    time2= np.linspace(0, len(signal2) / fs, num=len(signal2))
    a = plt.subplot(211)
    plt.plot(time, signal ,label='señal 1')
    plt.plot(time2, signal2 ,label='señal 2')
    plt.xlabel('Tiempo')  # Colocamos la etiqueta para el eje x
    plt.ylabel('Amplitud (unidad desconocida)')  # Colocamos la etiqueta para el eje y
    a.legend()
    a.set_ylim([-4000, 4000])
    b= plt.subplot(212)
    plt.plot(time2, signal2, label='señal2')
    plt.plot(time, signal,label='señal 1')
    plt.xlabel('Tiempo')  # Colocamos la etiqueta para el eje x
    plt.ylabel('Amplitud (unidad desconocida)')  # Colocamos la etiqueta para el eje y
    b.legend()
    b.set_ylim([-4000, 4000])
    plt.show()

#derecha
#@ray.remote
def ejecutar_doc1(tiempo):
    print("ejecuto 1")
    fs = 44100
    sd.default.samplerate = fs
    # print(fs)
    sd.default.channels = 1
    # derecho
    sd.default.device = (3,None)
    duration = 4
    value=duration * fs
    myrecording = sd.rec(int(value), samplerate=fs, channels=(1))
    sd.wait()
    print("grabando1")
    sd.stop()
    print("grabado1")
    sf.write('1_{}.wav'.format(tiempo), myrecording, fs)
#izquierda
#@ray.remote
def ejecutar_doc2(tiempo):
    print("ejecuto 2")
    fs = 44100
    sd.default.samplerate = fs
    # print(fs)
    sd.default.channels = 1
    duration = 4
    sd.default.device = (4,None)
    value2= duration * fs
    myrecording2 = sd.rec(int(value2), samplerate=fs, channels=(1))
    sd.wait()
    print("grabando2")
    sd.stop()
    print("grabado1")
    sf.write('2_{}.wav'.format(tiempo), myrecording2, fs)

def sonidos_detectados():
    myaudio = AudioSegment.from_wav("a1.wav")
    myaudio2 = AudioSegment.from_wav("b1.wav")
    silencio = silence.detect_nonsilent(myaudio, min_silence_len=1, silence_thresh=-32)
    silencio2 = silence.detect_nonsilent(myaudio2, min_silence_len=1, silence_thresh=-32)
    print(len(silencio))

if __name__ == '__main__':
    if (path.exists('nombres_diferencias.csv'))==False:
        print("Generando archivos necesarios")
        df = pd.DataFrame(columns=['nombres', 'distancias'])
        df.to_csv('nombres_diferencias.csv')
    if (path.exists('nombres_individuales.csv')) == False:
        print("Generando archivos necesarios")
        df = pd.DataFrame(columns=['nombres','distancias'])
        df.to_csv('nombres_individuales.csv')
    if (path.exists('nombres_valores.csv')) == False:
        print("Generando archivos necesarios")
        df = pd.DataFrame(columns=[''])
        df.to_csv('nombres_valores.csv')
    if (path.exists('mfcc.csv')) == False:
        print("Generando archivos necesarios")
        df = pd.DataFrame(columns=[''])
        df.to_csv('mfcc.csv')
    aux = False
    options = [1,2,3,4,5,6,7,8,88,89,9,10,11]
    de=6
    iz=7
    distancias=[0.5 ,1 ,1.5 ,2 ,2.5 ,3 ,3.5 ,4 ,4.5,5]
    a=b=c=d=e=f=g=h=i=j="0"
    while (aux==False):
        distancia = 0.
        warnings.simplefilter("ignore", DeprecationWarning)
        fs = 22050
        sd.default.samplerate = fs
        # print(fs)
        sd.default.channels = 1
        duration = 2
        print("1-Grabar")
        print("2-Ver señales ")
        print("3-ver fft y dispositivos")
        print("4-FFT")
        print("5-Ver señales transformadas")
        print("6-detectar sonidos")
        print("")
        print("88- csv")
        print("")
        print("")
        print("10-salir")

        error = True
        while error ==True:
            try:
                option = int(input())
                print("opcion: ", option)
                error = False
            except ValueError:
                        print("1-Grabar")
                        print("2-Ver señales ")
                        print("3-ver fft y dispositivos")
                        print("4-FFT")
                        print("5-Ver señales transformadas")
                        print("6-detectar sonidos")
                        print("")
                        print("88- csv")
                        print("")
                        print("")
                        print("10-salir")

        if option in options:


                if (option==1):
                    print("ingrese cantidad de repeteciones")
                    cantidad =1

                    distancia=-1
                    while (distancia > 500 or distancia < 0):
                        print("ingrese la distancia en centimetros")
                        distancia=int(input())

                    i=0
                    valores=[]
                    contador = 0
                    df = pd.read_csv("nombres_diferencias.csv", usecols=("nombres", "distancias"), dtype=str)

                    while i < cantidad:

                        tiempo = int(time.time() )
                        p1 = Process(target=ejecutar_doc1, args=(tiempo,))
                        p1.start()
                        p2 = Process(target=ejecutar_doc2, args=(tiempo,))
                        p2.start()
                        p1.join()
                        p2.join()
                        i= i+1
                        time.sleep(1)
                        valores.append([tiempo, distancia])
                    df1 = pd.DataFrame(valores, columns=("nombres", "distancias"))
                    print("hey")
                    df = df.append(df1, ignore_index=True)
                    print(df)
                    df.to_csv("nombres_diferencias.csv")
                    input()
                if (option == 2):
                        a= pd.read_csv('nombres_diferencias.csv')
                        valores = a['nombres']
                        for i in range(len(valores)):
                            uno=str("1_"+str(valores[i])+".wav")
                            dos=str("2_"+str(valores[i])+".wav")

                            print(uno,dos)
                            archivo = wave.open(uno, 'rb')
                            archivo2 = wave.open(dos, 'rb')
                            canales = archivo.getnchannels()
                            frames = archivo.getframerate()
                            fs = frames
                            datos = archivo.getparams()
                            samples = archivo.getsampwidth()
                            data = archivo.readframes(-1)
                            data2 = archivo2.readframes(-1)
                            signal = np.frombuffer(data, 'Int16')
                            signal2 = np.frombuffer(data2, 'Int16')
                            show(signal, signal2)
                            print("presiona enter para continuar")
                        input()
                if option == 3:
                    print(sd.query_devices())
                    a = pd.read_csv('nombres_diferencias.csv')
                    valores = a['nombres']
                    #print("ingrese frecuencia")
                    #frec=int(input())
                    for i in range(len(valores)):
                        uno = str("1_" + str(valores[i]) + ".wav")
                        dos = str("2_" + str(valores[i]) + ".wav")
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        W = np.fft.fftfreq(len(signal))*44100
                        val=W
                        fft_signal = np.fft.fft(signal)
                        fft_theo=(2.0*np.abs(fft_signal/len(signal)))
                        cut_f_signal = fft_signal.copy()
                        #PARA VER FRECUENCIA DE SONIDO FFT
                        plt.plot((W), (((cut_f_signal)/44100)))
                        print(cut_f_signal/44100)
                        plt.xlabel('Frecuencia Hz')  # Colocamos la etiqueta para el eje x
                        plt.ylabel('Cantidad de muestras')  # Colocamos la etiqueta para el eje y


                        axes = plt.gca()

                        axes.set_xlim([0,500])

                        plt.show()
                    print("presiona enter para continuar")
                    input()
                if option == 4:
                    a = pd.read_csv('nombres_diferencias.csv')
                    valores = a['nombres']
                    print("ingrese frecuencia")
                    frec=int(input())
                    for i in range(len(valores)):
                        uno = str("1_" + str(valores[i]) + ".wav")

                        dos = str("2_" + str(valores[i]) + ".wav")
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        print(len(signal))

                        W = np.fft.fftfreq(len(signal))*44100
                        val=W
                        fft_signal = np.fft.fft(signal)
                        fft_theo=(2.0*np.abs(fft_signal/len(signal)))
                        cut_f_signal = fft_signal.copy()
                        #  PARA VER FRECUENCIA DE SONIDO FFT
                        #plt.plot(W,cut_f_signal)
                        #plt.show()
                        a=frec-5
                        b=frec+5
                        for i in range(0,len(W)):
                            if((W[i]<-b) or (W[i])>b):
                                cut_f_signal[[i]]=0
                            else:
                                if((W[i]>-a)  and   (W[i]<a)):
                                    cut_f_signal[[i]]=0

                        final = np.fft.ifft(cut_f_signal)
                        final = final.astype('int16')
                        for inicio in range(0, 1000):
                            final[inicio] = 10
                        for fin in range(len(signal) - 1000, len(signal)):
                            final[fin] = 10
                        wavfile.write('fft_{}'.format(uno), 44100, final)
                        signal = np.frombuffer(data2, 'Int16')
                        fft_signal = np.fft.fft(signal)
                        W = np.fft.fftfreq(len(signal))*44100
                        val = W > 0
                        fft_signal = np.fft.fft(signal)
                        cut_f_signal = fft_signal.copy()
                        a=frec-10
                        b=frec+10
                        for i in range(0,len(W)):
                            if((W[i]<-b) or (W[i])>b):
                                cut_f_signal[[i]]=0
                            else:
                                if((W[i]>-a)  and   (W[i]<a)):
                                    cut_f_signal[[i]]=0
                        plt.xlabel('Frecuencia (Hz)')  # Colocamos la etiqueta para el eje x
                        plt.ylabel('Amplitud (unidad desconocida)')  # Colocamos la etiqueta para el eje y

                        # plt.show()
                        # plt.plot(np.fft.ifft(cut_f_signal))
                        plt.xlabel('Numero de muestra')  # Colocamos la etiqueta para el eje x
                        plt.ylabel('Amplitud (unidad desconocida)')  # Colocamos la etiqueta para el eje y

                        # plt.show()
                        # cut_f_signal[(W < 0.0039)] = 0
                        # cut_f_signal[(W > 0.006)] = 0
                        final = np.fft.ifft(cut_f_signal)
                        final = final.astype('int16')
                        for inicio in range(0, 1000):
                            final[inicio] = 10
                        for fin in range(len(signal) - 1000, len(signal)):
                            final[fin] = 10
                        wavfile.write('fft_{}'.format(dos), 44100, final)
                if option==5:
                    a = pd.read_csv('nombres_diferencias.csv')
                    valores = a['nombres']
                    for i in range(len(valores)):
                        uno = str("fft_1_" + str(valores[i]) + ".wav")
                        dos = str("fft_2_" + str(valores[i]) + ".wav")
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')
                        show(signal,signal2)
                        ''' ESPECTROGRAMA

                        (rate1, sig1) = wav.read(uno)
                        (rate2, sig2) = wav.read(dos)
                        f, t, Sxx = sg.spectrogram(signal, rate1, nperseg=441)
                        f2, t2, Sxx2 = sg.spectrogram(signal2, rate2, nperseg=441)
                        a = plt.subplot(211)
                        plt.pcolor(t, f, Sxx, vmin=0, vmax=2000, cmap='gist_earth')
                        axes = plt.gca()
                        axes.set_ylim([-100, 7000])
                        plt.ylabel('Frecuencia [Hz]')
                        plt.xlabel('Tiempo [sec]')
                        b = plt.subplot(212)
                        plt.pcolor(t2, f2, Sxx2, vmin=0, vmax=2000, cmap='gist_earth')
                        axes = plt.gca()
                        axes.set_ylim([-100, 7000])
                        plt.ylabel('Frecuencia [Hz]')
                        plt.xlabel('Tiempo [sec]')
                        plt.show()
                        '''
                    print("presiona enter para continuar")
                    input()
                if (option==6):
                    a = pd.read_csv('nombres_diferencias.csv')
                    valores = a['nombres']
                    distancia = a['distancias']
                    dif1 = ""
                    cont = 0
                    print("ingrese duracion del sonido   ej: 0.1")
                    dur = float(input())
                    dur2 = dur
                    dur = int(dur/0.1)
                    #a = int(input())

                    cont_name = 0

                    b = pd.read_csv('nombres_individuales.csv')
                    name_csv = []
                    dif_csv = []
                    df1 = pd.DataFrame([], columns=['nombres', 'distancias'])
                    for i in range(len(valores)):
                        #print("sonido? a)alto b)bajo")

                        #mx = str(input())
                        mx = "a"
                        if mx == "a":
                            maximo = 2004730
                        else:
                            maximo = 68000
                        uno = str("fft_1_" + str(valores[i]) + ".wav")
                        dos = str("fft_2_" + str(valores[i]) + ".wav")
                        print("nombres de los archivos: ", uno, dos)
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')
                        muestra = []
                        aux = 0
                        suma_ant = 0
                        suma_ant2 = 0
                        contador=0

                        for j in range(0,len(signal)-int(4410*dur),int(4410*dur)):
                            suma = 0
                            suma2 = 0

                            for x in range(0,4410*dur):
                                suma=np.abs(suma)+np.abs(signal[j+x])
                                suma2=np.abs(suma2)+np.abs(signal2[j+x])
                            print("-")

                            print(suma/100000)
                            print(suma_ant/100000)

                            print(suma2/100000)
                            print(suma_ant2/100000)

                            print(maximo/100000)
                            print("-")
                            if((suma > maximo) and (suma_ant > suma)):
                                print("ooeeeeee")
                            if ((suma2 > maximo) and (suma_ant2 >suma2 )):
                                print("Aaaa")
                            if ( (((suma > maximo) and (suma_ant < suma)) or ((suma2 > maximo) and (suma_ant2 <suma2 )))):
                                print("oe")
                                muestra.append(j)
                                print(muestra)
                                    #print("x1", (j+x)/4410)
                                #print(suma_ant)
                            suma_ant=suma
                            suma_ant2=suma2
                            #print(suma, suma2, maximo)
                        #print(muestra)
                        #if (mx=="b"):
                        #    for z in range(0, int(len(muestra))-1):
                        #       print((muestra[z-1])/1000,(muestra[z])/1000,(muestra[z+1])/1000)
                        print(len(muestra))
                        for valor in range(0, int(len(muestra))):
                            print(muestra[valor] / 44.1)
                            if(((((len(muestra)==1 and muestra[valor]/ 44.1) - 441 / 2) >0) )):
                                for m in range((muestra[valor])-dur*4410,  (muestra[valor]+dur*4410), dur*441):
                                    print("oe")
                                    center= 0
                                    center2=0
                                    for n in range (0,dur*441):
                                        center=np.abs(center)+np.abs(signal[m+n])
                                        center2=np.abs(center2)+np.abs(signal2[m+n])


                                print(center)
                                tiempo = ((muestra[valor]) -9700)
                                tiempo2 = ((muestra[valor]) + 5000)
                                if (tiempo>0 and tiempo2>0):
                                    tiempo_med=np.max(signal[tiempo:tiempo2])
                                    tiempo_med2=np.max(signal2[tiempo:tiempo2])
                                    if(tiempo_med >= tiempo_med2 ):
                                       signal3 = signal[tiempo:tiempo2].tolist()
                                       c = signal3.index(tiempo_med)
                                       tiempo2 = tiempo + c  +(4500)
                                       tiempo = tiempo  +c   -(3000)

                                    if tiempo_med2>tiempo_med:
                                       signal3=signal2[tiempo:tiempo2].tolist()
                                       c=signal3.index(tiempo_med2)
                                       tiempo2 = tiempo + c  +(4500)
                                       tiempo=tiempo+  c   -(3000)

                                    tiempo = tiempo/44.1
                                    tiempo2 = tiempo2/44.1
                                    t1 = tiempo - 100 * dur2 *10
                                    t2 = tiempo2 + 60 * dur2*10
                                    t3 = tiempo - 100 *dur2*10
                                    t4 = tiempo2 + 60 *dur2*10
                                    newAudio = AudioSegment.from_wav(uno)
                                    newAudio = newAudio[t1:t2]
                                    name = str(cont_name)
                                    dif1 = str(distancia[i])
                                    newAudio.export('{}.a.wav'.format(name), format="wav")
                                    newAudio = AudioSegment.from_wav(dos)
                                    newAudio = newAudio[t3:t4]
                                    newAudio.export('{}.b.wav'.format(name), format="wav")
                                    name1 = (str(cont_name) + ".a")
                                    name_csv.append([name1, dif1])
                                    name2 = (str(cont_name) + ".b")
                                    name_csv.append([name2, dif1])
                                    cont_name = cont_name + 1




                            if(((((muestra[valor]) / 44.1) - 441 / 2) >0 and valor!=0 and valor!=len(muestra) and (((muestra[valor]-muestra[valor-1*dur]>(4410*2  ))and (muestra[valor+1]-muestra[valor]>(4410*3)))or muestra[valor-1]==0))):
                               for m in range((muestra[valor])-dur*4410,  (muestra[valor]+dur*4410), dur*441):

                                 print("oe")
                                 center= 0
                                 center2=0
                                 for n in range (0,dur*441):
                                     center=np.abs(center)+np.abs(signal[m+n])
                                     center2=np.abs(center2)+np.abs(signal2[m+n])

                               print(center)
                               tiempo = ((muestra[valor]) -9700)
                               tiempo2 = ((muestra[valor]) + 5000)
                               if (tiempo>0 and tiempo2>0):
                                    tiempo_med=np.max(signal[tiempo:tiempo2])
                                    tiempo_med2=np.max(signal2[tiempo:tiempo2])





                                    if(tiempo_med >= tiempo_med2 ):
                                        signal3 = signal[tiempo:tiempo2].tolist()
                                        c = signal3.index(tiempo_med)
                                        tiempo2 = tiempo + c  +(4500)
                                        tiempo = tiempo  +c   -(3000)

                                    if tiempo_med2>tiempo_med:
                                        signal3=signal2[tiempo:tiempo2].tolist()
                                        c=signal3.index(tiempo_med2)
                                        tiempo2 = tiempo + c  +(4500)
                                        tiempo=tiempo+  c   -(3000)

                                    tiempo = tiempo/44.1
                                    tiempo2 = tiempo2/44.1
                                    t1 = tiempo - 100 * dur2 *10
                                    t2 = tiempo2 + 60 * dur2*10
                                    t3 = tiempo - 100 *dur2*10
                                    t4 = tiempo2 + 60 *dur2*10
                                    newAudio = AudioSegment.from_wav(uno)
                                    newAudio = newAudio[t1:t2]
                                    name = str(cont_name)
                                    dif1 = str(distancia[i])
                                    newAudio.export('{}.a.wav'.format(name), format="wav")
                                    newAudio = AudioSegment.from_wav(dos)
                                    newAudio = newAudio[t3:t4]
                                    newAudio.export('{}.b.wav'.format(name), format="wav")
                                    name1 = (str(cont_name) + ".a")
                                    name_csv.append([name1, dif1])
                                    name2 = (str(cont_name) + ".b")
                                    name_csv.append([name2, dif1])
                                    cont_name = cont_name + 1
                    df1 =pd.DataFrame(name_csv, columns=("nombres","distancias"))
                    df1.to_csv("nombres_individuales.csv")

                if option==7:
                    a = pd.read_csv('nombres_individuales.csv')
                    valores = a['nombres']
                    for i in range(0,len(valores),2):
                        uno = str(str(valores[i])+".wav")
                        #print(uno)
                        dos = str(str(valores[i+1])+".wav")
                        #print(dos)
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')
                        plt.subplot(211)
                        plt.title(str(valores[i]))
                        plt.plot(signal)
                        plt.plot(signal2)
                        axes = plt.gca()
                        plt.subplot(212)
                        plt.title(str(valores[i+1]))
                        plt.plot(signal2)
                        plt.plot(signal)
                        axes = plt.gca()
                        plt.show()

                if option==8:
                    a = pd.read_csv('nombres_individuales.csv')
                    valores = a['nombres']
                    for i in range(0,len(valores),2):
                        uno = str(str(valores[i])+".wav")
                        #print(uno)
                        dos = str(str(valores[i+1])+".wav")
                        #print(dos)
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')
                        f, t, Sxx = sg.spectrogram(signal, fs, nperseg=441)

                        plt.pcolormesh(t, f, Sxx)
                        axes = plt.gca()
                        axes.set_ylim([200, 2050])
                        plt.ylabel('Frecuencia [Hz]')
                        plt.xlabel('Tiempo [sec]')
                        plt.show()
                        f2, t2, Sxx2 = sg.spectrogram(signal2, fs, nperseg=441)

                        plt.pcolormesh(t2, f2, Sxx2)
                        axes = plt.gca()
                        axes.set_ylim([200, 2050])
                        plt.ylabel('Frecuencia [Hz]')
                        plt.xlabel('Tiempo [sec]')
                        plt.show()

                if option == 9:
                    a = pd.read_csv('nombres_individuales.csv')
                    valores = a['nombres']
                    for i in range(0, len(valores), 2):
                        uno = str(str(valores[i]) + ".wav")
                        print(uno)
                        dos = str(str(valores[i + 1]) + ".wav")
                        print(dos)
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')

                        plt.subplot(211)
                        plt.title("señales")

                        plt.plot(signal)
                        plt.plot(signal2)

                        # axes = plt.gca()
                        # axes.set_ylim([-700, 700])


                        # axes = plt.gca()
                        # axes.set_ylim([-700, 700])
                        plt.subplot(212)
                        plt.title('Convolve')

                        x= fftconvolve(signal,signal2,"same")
                        plt.plot(x)
                        plt.show()
                if option==88:
                    print("cual?")
                    ingresado=input()
                    a = pd.read_csv('nombres_individuales.csv')
                    print(a)
                    valores = a['nombres']
                    valores_csv = []
                    distancia =a['distancias']
                    #print(distancia)
                    #print(len(valores))
                    #for i in range(0, int(len(valores)),2):
                    for i in range(0, int(len(valores)),2):

                        dis=int(distancia[i]/10)*10
                        string= int(distancia[i])
                        array = np.zeros(50,dtype=int)
                        array[int(string/10)]=1


                        uno = str(str(valores[i])+".wav")
                        dos = str(str(valores[i+1]) + ".wav")
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')
                        #print(signal)
                        #print(signal2)

                        #plt.plot(signal)
                        #plt.plot(signal2)
                        #plt.show()
                        #convolve = fftconvolve(signal, signal2, "same") / 10
                        #convolve = np.round(convolve)
                        #print(len(convolve))
                        #if len(convolve)<9041:
                        #    np.append(convolve,[0])

                        #opcion 1
                        """
                        if np.max(signal) < np.max(signal2):
                            signal_mayor = signal
                        else:
                            signal_mayor = signal2
                        """
                        #opcion 2
                        signal_mayor = fftconvolve(signal, signal2, "same")
                        signal_mayor = np.round(signal_mayor)

                        diferencia=np.max(signal)-np.max(signal2)

                        prim = []
                        prim2 = []
                        prim3 = []
                        prim4 = []
                        prim5 = []
                        prim6 = []
                        prim7 = []
                        prim8 = []
                        prim9 = []
                        prim10 = []
                        prim11 = []
                        prim12 = []
                        prim13 = []
                        prim14 = []
                        prim15 = []
                        prim16 = []
                        prim17 = []
                        prim18 = []
                        prim19 = []
                        prim20 = []
                        prim21 = []
                        prim22 = []
                        prim23 = []
                        prim24 = []
                        prim25 = []
                        prim26 = []
                        prim27 = []
                        prim28 = []
                        prim29 = []
                        prim30= []
                        print("size original", len(signal))
                        for x in range(0, len(signal)-29, 30):
                            prim.append(int(signal_mayor[x]))
                            prim2.append(int(signal_mayor[x+1]))
                            prim3.append(int(signal_mayor[x+2]))
                            prim4.append(int(signal_mayor[x+3]))
                            prim5.append(int(signal_mayor[x+4]))
                            prim6.append(int(signal_mayor[x+5]))
                            prim7.append(int(signal_mayor[x+6]))
                            prim8.append(int(signal_mayor[x+7]))
                            prim9.append(int(signal_mayor[x+8]))
                            prim10.append(int(signal_mayor[x+9]))
                            prim11.append(int(signal_mayor[x+10]))
                            prim12.append(int(signal_mayor[x+11]))
                            prim13.append(int(signal_mayor[x+12]))
                            prim14.append(int(signal_mayor[x+13]))
                            prim15.append(int(signal_mayor[x+14]))
                            prim16.append(int(signal_mayor[x+15]))
                            prim17.append(int(signal_mayor[x+16]))
                            prim18.append(int(signal_mayor[x+17]))
                            prim19.append(int(signal_mayor[x+18]))
                            prim20.append(int(signal_mayor[x+19]))
                            prim22.append(int(signal_mayor[x + 21]))
                            prim21.append(int(signal_mayor[x+20]))
                            prim23.append(int(signal_mayor[x + 22]))
                            prim24.append(int(signal_mayor[x + 23]))
                            prim25.append(int(signal_mayor[x + 24]))
                            prim26.append(int(signal_mayor[x + 25]))
                            prim27.append(int(signal_mayor[x + 26]))
                            prim28.append(int(signal_mayor[x + 27]))
                            prim29.append(int(signal_mayor[x + 28]))
                            prim30.append(int(signal_mayor[x + 29]))
                        plt.plot(prim)
                        plt.show()
                        prim=((mfcc(np.asarray(prim[0:len(prim)]), int(1000*np.log10(len(signal))), nfft=1103)*10).astype(int)).transpose().tolist()
                        plt.plot(prim[0])
                        plt.show()
                        prim2=((mfcc(np.asarray(prim2[0:len(prim2)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim3=((mfcc(np.asarray(prim3[0:len(prim3)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim4=((mfcc(np.asarray(prim4[0:len(prim4)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim5=((mfcc(np.asarray(prim5[0:len(prim5)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim6=((mfcc(np.asarray(prim6[0:len(prim6)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim7=((mfcc(np.asarray(prim7[0:len(prim7)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim8=((mfcc(np.asarray(prim8[0:len(prim8)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim9=((mfcc(np.asarray(prim9[0:len(prim9)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim10=((mfcc(np.asarray(prim10[0:len(prim10)]), 44100/30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim12=((mfcc(np.asarray(prim12[0:len(prim12)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim13=((mfcc(np.asarray(prim13[0:len(prim13)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim11=((mfcc(np.asarray(prim11[0:len(prim11)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim14=((mfcc(np.asarray(prim14[0:len(prim14)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim15=((mfcc(np.asarray(prim15[0:len(prim15)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim16=((mfcc(np.asarray(prim16[0:len(prim16)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim17=((mfcc(np.asarray(prim17[0:len(prim17)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim18=((mfcc(np.asarray(prim18[0:len(prim18)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim19=((mfcc(np.asarray(prim19[0:len(prim19)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim20=((mfcc(np.asarray(prim20[0:len(prim20)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim21=((mfcc(np.asarray(prim21[0:len(prim21)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim22=((mfcc(np.asarray(prim22[0:len(prim22)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim23=((mfcc(np.asarray(prim23[0:len(prim23)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim24=((mfcc(np.asarray(prim24[0:len(prim24)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim25=((mfcc(np.asarray(prim25[0:len(prim25)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim26=((mfcc(np.asarray(prim26[0:len(prim26)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim27=((mfcc(np.asarray(prim27[0:len(prim27)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim28=((mfcc(np.asarray(prim28[0:len(prim28)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim29=((mfcc(np.asarray(prim29[0:len(prim29)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()
                        prim30=((mfcc(np.asarray(prim30[0:len(prim30)]), 44100 / 30, nfft=1103)*10).astype(int)).transpose().tolist()

                        prim = prim[0]
                        prim2 = prim2[0]
                        prim3 = prim3[0]
                        prim4 = prim4[0]
                        prim5 = prim5[0]
                        prim6 = prim6[0]
                        prim7 = prim7[0]
                        prim8 = prim8[0]
                        prim9 = prim9[0]
                        prim10 = prim10[0]
                        prim11 = prim11[0]
                        prim12 = prim12[0]
                        prim13 = prim13[0]
                        prim14 = prim14[0]
                        prim15 = prim15[0]
                        prim16 = prim16[0]
                        prim17 = prim17[0]
                        prim18 = prim18[0]
                        prim19 = prim19[0]
                        prim20 = prim20[0]
                        prim21 = prim21[0]
                        prim22 = prim22[0]
                        prim23 = prim23[0]
                        prim24 = prim24[0]
                        prim25 = prim25[0]
                        prim26 = prim26[0]
                        prim27 = prim27[0]
                        prim28 = prim28[0]
                        prim29 = prim29[0]
                        prim30 = prim30[0]

                        prim.append(diferencia)
                        prim2.append(diferencia)
                        prim3.append(diferencia)
                        prim4.append(diferencia)
                        prim5.append(diferencia)
                        prim6.append(diferencia)
                        prim7.append(diferencia)
                        prim8.append(diferencia)
                        prim9.append(diferencia)
                        prim10.append(diferencia)
                        prim11.append(diferencia)
                        prim12.append(diferencia)
                        prim13.append(diferencia)
                        prim14.append(diferencia)
                        prim15.append(diferencia)
                        prim16.append(diferencia)
                        prim17.append(diferencia)
                        prim18.append(diferencia)
                        prim19.append(diferencia)
                        prim20.append(diferencia)
                        prim21.append(diferencia)
                        prim22.append(diferencia)
                        prim23.append(diferencia)
                        prim24.append(diferencia)
                        prim25.append(diferencia)
                        prim26.append(diferencia)
                        prim27.append(diferencia)
                        prim28.append(diferencia)
                        prim29.append(diferencia)
                        prim30.append(diferencia)


                        if ingresado == "b":
                            prim.append(dis)
                            prim2.append(dis)
                            prim3.append(dis)
                            prim4.append(dis)
                            prim5.append(dis)
                            prim6.append(dis)
                            prim7.append(dis)
                            prim8.append(dis)
                            prim9.append(dis)
                            prim10.append(dis)
                            prim11.append(dis)
                            prim12.append(dis)
                            prim13.append(dis)
                            prim14.append(dis)
                            prim15.append(dis)
                            prim16.append(dis)
                            prim17.append(dis)
                            prim18.append(dis)
                            prim19.append(dis)
                            prim20.append(dis)
                            prim21.append(dis)
                            prim22.append(dis)
                            prim23.append(dis)
                            prim24.append(dis)
                            prim25.append(dis)
                            prim26.append(dis)
                            prim27.append(dis)
                            prim28.append(dis)
                            prim29.append(dis)
                            prim30.append(dis)

                        if ingresado == "a":
                            for z in range(0,20):
                                prim.append(array[z])
                                prim2.append(array[z])
                                prim3.append(array[z])
                                prim4.append(array[z])
                                prim5.append(array[z])
                                prim6.append(array[z])
                                prim7.append(array[z])
                                prim8.append(array[z])
                                prim9.append(array[z])
                                prim10.append(array[z])
                                prim11.append(array[z])
                                prim12.append(array[z])
                                prim13.append(array[z])
                                prim14.append(array[z])
                                prim15.append(array[z])
                                prim16.append(array[z])
                                prim17.append(array[z])
                                prim18.append(array[z])
                                prim19.append(array[z])
                                prim20.append(array[z])
                                prim21.append(array[z])
                                prim22.append(array[z])
                                prim23.append(array[z])
                                prim24.append(array[z])
                                prim25.append(array[z])
                                prim26.append(array[z])
                                prim27.append(array[z])
                                prim28.append(array[z])
                                prim29.append(array[z])
                                prim30.append(array[z])
                        valores_csv.append(prim)
                        valores_csv.append(prim2)
                        valores_csv.append(prim3)
                        valores_csv.append(prim4)
                        valores_csv.append(prim5)
                        valores_csv.append(prim6)
                        valores_csv.append(prim7)
                        valores_csv.append(prim8)
                        valores_csv.append(prim9)
                        valores_csv.append(prim10)
                        valores_csv.append(prim12)
                        valores_csv.append(prim13)
                        valores_csv.append(prim14)
                        valores_csv.append(prim15)
                        valores_csv.append(prim16)
                        valores_csv.append(prim17)
                        valores_csv.append(prim18)
                        valores_csv.append(prim19)
                        valores_csv.append(prim20)
                        valores_csv.append(prim21)
                        valores_csv.append(prim22)
                        valores_csv.append(prim23)
                        valores_csv.append(prim24)
                        valores_csv.append(prim25)
                        valores_csv.append(prim26)
                        valores_csv.append(prim27)
                        valores_csv.append(prim28)
                        valores_csv.append(prim29)
                        valores_csv.append(prim30)



                    df1 = pd.DataFrame( valores_csv[:])
                    print(df1)
                    df1.to_csv("nombres_valores.csv")


                if option == 89:
                    a = pd.read_csv('nombres_individuales.csv')
                    valores = a['nombres']
                    valores_csv = []
                    distancia = a['distancias']
                    cont=0
                    #print(len(valores))

                    for i in range(0, int(len(valores)),2):

                        dis = int(distancia[i])
                        string = int(distancia[i])
                        array = np.zeros(50, dtype=int).tolist()
                        array[int(string / 10)] = 1
                        #print(type(array))
                        uno = str(str(valores[i]) + ".wav")
                        dos = str(str(valores[i + 1]) + ".wav")
                        y, sr = librosa.load(uno)
                        y2, sr2 = librosa.load(dos)

                        F,f_names=audioFeatureExtraction.stFeatureExtraction(y, sr, 0.050*sr, 0.025*sr)
                        F2,f_names2=audioFeatureExtraction.stFeatureExtraction(y2, sr2, 0.050*sr2, 0.025*sr2)
                        a=F[0,:].tolist()
                        a1=F[1,:].tolist()
                        a=a+a1
                        b=F2[0,:].tolist()
                        b1=F2[1,:].tolist()
                        b=b+b1
                        #print(a)
                        #print(len(a))
                        #print(f_names[0])
                        #plt.plot(librosa.feature.melspectrogram(y))
                        #plt.show()

                        #print(len(sig1)/rate1)
                        #(rate2, sig2) = wav.read(dos)

                        #mfcc_feat1 = mfcc(sig1, 44100,nfft=2046)
                        #print("mfcc",(  len(mfcc_feat1[1,:]) ))
                        #mfcc_feat2 = mfcc(sig2, 44100,nfft=2046)
                        #fbank_feat1 = logfbank(sig1, rate1,nfft=2046)
                        #print("fbank", len(fbank_feat1[1,:]))
                        #fbank_feat2 = logfbank(sig2, rate2,nfft=2046)
                        #features1 = np.concatenate((mfcc_feat1, fbank_feat1), axis=1)
                        #features2 = np.concatenate((mfcc_feat2, fbank_feat2), axis=1)
                        #plt.imshow(fbank_feat1)
                        #plt.show()
                        #print("f1",features1)
                        #print("f2",features2)
                        #print("fbank",fbank_feat2)
                        #a=(fbank_feat1[1,:]).tolist()
                        #b=(fbank_feat2[1,:]).tolist()
                        #a=(mfcc_feat1[1,:]).tolist()
                        #a1=(mfcc_feat[2,:]).tolist()
                        #a2=(mfcc_feat[3,:]).tolist()
                        #a=(features1[1,:]).tolist()
                        #b=(features2[1,:]).tolist()
                        c=a+b+array
                        #b=(mfcc_feat2[1,:]).tolist()
                        #b1=(mfcc_feat2[2,:]).tolist()
                        #b2=(mfcc_feat2[3,:]).tolist()
                        #a=a+a1+a2
                        #b=b+b1+b2
                        #c=a+b
                        #d=c+array
                        valores_csv.append(c)


                    df1 = pd.DataFrame(valores_csv[:])
                    df1.to_csv("mfcc.csv")

                if option == 10:
                    aux= True
                if option == 11:
                    a = pd.read_csv('nombres_diferencias.csv')
                    valores = a['nombres']
                    uno = str("1_" + str(1559259194) + ".wav")
                    dos = str("2_" + str(1559259194) + ".wav")
                    archivo = wave.open(uno, 'rb')
                    archivo2 = wave.open(dos, 'rb')
                    canales = archivo.getnchannels()
                    frames = archivo.getframerate()
                    fs = frames
                    datos = archivo.getparams()
                    samples = archivo.getsampwidth()
                    data = archivo.readframes(-1)
                    data2 = archivo2.readframes(-1)
                    signal = np.frombuffer(data, 'Int16')
                    signal2 = np.frombuffer(data2, 'Int16')
                    fft_signal = np.fft.fft(signal)
                    W = np.fft.fftfreq(len(signal))
                    cut_f_signal = fft_signal.copy()
                    cut_f_signal[(W < 0.01)] = 0
                    cut_f_signal[(W > 0.2)] = 0
                    final = np.fft.ifft(cut_f_signal)

                    plt.plot(fft_signal)
                    plt.show()

        else:
            print("opcion no valida")
            print("presiona enter")
            input()

        # silencio = silence.detect_silence(myaudio, min_silence_len=1, silence_thresh=16)

        # silencio2 = silence.detect_nonsilent(myaudio, min_silence_len=1, silence_thresh=-30)
        # silencio = [((start/1000),(stop/1000)) for start,stop in silencio] #convert to sec
        # print(silencio2)
        # print(silencio[0][1])
        # print(silencio[0][1])
        # print(silencio2[0][1])

    #p1 = Process(target=ejecutar_doc1())
    #p1.start()
    #p2 = Process(target=ejecutar_doc2)
    #p2.start()
    #p1.join()
    #p2.join()

    #hilo1 = threading.Thread(target=ejecutar_doc1(),daemon=True)
    #hilo2 = threading.Thread(target=ejecutar_doc2(),daemon=True)

    #hilo1.start()
    #hilo2.start()









    # graba simultaneamente 2 microfonos
