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


def fft_graph(data,data2):
    signal = np.frombuffer(data, 'Int16')
    signal2 = np.frombuffer(data2, 'Int16')
    fft_signal =np.fft.fft(signal)
    fft_t= 2*np.abs(fft_signal / 88200)
    freq = np.fft.fftfreq(len(signal))
    mask = freq > 0
    W = np.fft.fftfreq(len(signal))
    cut_f_signal = fft_signal.copy()
    cut_f_signal[(W < 0.01)] = 0
    cut_f_signal[(W > 0.2)] = 0
    # cut_f_signal[(W>4.0   )] = 0
    final = np.fft.ifft(cut_f_signal)

    fft_signal2 =np.fft.fft(signal2)
    fft_t2= 2*np.abs(fft_signal2 / 88200)
    freq2 = np.fft.fftfreq(len(signal2))
    mask2 = freq2 > 0
    W2= np.fft.fftfreq(len(signal2))
    cut_f_signal2 = fft_signal2.copy()
    cut_f_signal2[(W < 0.01)] = 0
    cut_f_signal2[(W > 0.1)] = 0
    final2 = np.fft.ifft(cut_f_signal2)
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    time2 = np.linspace(0, len(signal2) / fs, num=len(signal2))
    plt.subplot(211)
    plt.plot(time, final)
    plt.subplot(212)
    plt.plot(time2, final2)
    plt.show()

def fft_hamming(data):

    signal_no_h = np.frombuffer(data, 'Int16')
    ventaneo = len(signal_no_h)/200
    hamming= np.hamming(ventaneo)
    plt.plot(signal_no_h)
    plt.show()
    signal=signal_no_h.copy()
    for i in range(0,88074,378):
        signal[i:(i+441)] = signal_no_h[i:(i+441)] * hamming
    plt.plot(signal)
    plt.show()
    fft_signal =np.fft.fft(signal)
    fft_t= 2*np.abs(fft_signal / 88200)
    freq = np.fft.fftfreq(len(signal))
    mask = freq > 0
    W = np.fft.fftfreq(len(signal))
    cut_f_signal = fft_signal.copy()
    cut_f_signal[(W < 0.01)] = 0
    cut_f_signal[(W > 0.2)] = 0
    #cut_f_signal[(W>4.0   )] = 0
    final = np.fft.ifft(cut_f_signal)
    f, (signal_graph, signal_graph2) = plt.subplots(1, 2, )
    signal_graph.plot(freq[mask], fft_t[mask])
    signal_graph2.plot(final)
    plt.show()
    #final=final.astype('int16')
    #wavfile.write(name, 44100, final)
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
        print("2-Ver señales y espectrogramas")
        print("3-ver fft y dispositivos")
        print("4-Transformada de fourier FFT")
        print("5-Ver selñales transformadas")
        print("6-buscar sonidos en señal")
        print("7-Ver sonidos obtenidos")
        print("-")
        print("")
        print("89-pasar MFCC a CSV")
        print("10-salir")

        error = True
        while error ==True:
            try:
                option = int(input())
                print("opcion: ", option)
                error = False
            except ValueError:
                print("Error!")
                print("Ingrese un entero")
                print("1-Grabar")
                print("2-Ver señales y espectrogramas")
                print("3-ver fft y dispositivos")
                print("4-Transformada de fourier FFT")
                print("5-Ver selñales transformadas")
                print("6-buscar sonidos en señal")
                print("-")
                print("")
                print("89-pasar MFCC a CSV")
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
                        print("ingrese duracion")
                        #dur=float(input())
                        dur=0.1
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
                            (rate1, sig1) = wav.read(uno)
                            (rate2, sig2) = wav.read(dos)
                            f, t, Sxx = sg.spectrogram(signal,fs,nperseg=441)
                            f2, t2, Sxx2 = sg.spectrogram(signal2,fs,nperseg=441)
                            a = plt.subplot(211)
                            plt.pcolor(t, f, Sxx, vmax=10, cmap='Greens')
                            axes = plt.gca()
                            axes.set_ylim([200, 1000])
                            plt.ylabel('Frecuencia [Hz]')
                            plt.xlabel('Tiempo [sec]')
                            b = plt.subplot(212)
                            plt.pcolor(t2, f2, Sxx2, vmax=10, cmap='Greens')
                            axes = plt.gca()
                            axes.set_ylim([200, 1000])
                            plt.ylabel('Frecuencia [Hz]')
                            plt.xlabel('Tiempo [sec]')
                            plt.show()


                            mayor=0
                            array=[]
                            for i in Sxx:
                                sum=0
                                for n in i:
                                    sum+=n
                                array.append(sum)
                            i=0
                            while (i<3):
                                array[i] = 0
                                i=i+1
                            print(("ingrese maximo:"))
                            #may=int(input())
                            may=110
                            a=array.index(max(array))
                            columna = f[a]
                            detectado=Sxx[a]
                            print(columna, len(detectado))
                            i=0
                            cont=0
                            while(i<len(detectado)-7):

                                if(detectado[i]<may-50 and detectado[i+1]>may and detectado[i+5]>may):
                                    print(i*0.009)
                                    cont=cont+1
                                    i=i+(5)
                                else:
                                    i=i+2
                            print("detectados: ", cont)
                            mayor=0
                            array=[]
                            for i in Sxx2:
                                sum=0
                                for n in i:
                                    sum+=n
                                array.append(sum)
                            i=0
                            while (i<3):
                                array[i] = 0
                                i=i+1
                            a=array.index(max(array))
                            columna = f2[a]
                            detectado=Sxx2[a]
                            print(columna, len(detectado))
                            i=10
                            cont=0
                            while(i<len(detectado)-7):
                                if(detectado[i]<may-50 and detectado[i+1]>may and detectado[i+5]>may):
                                    print(i*0.009)
                                    cont=cont+1
                                    i=i+(5)
                                else:
                                    i=i+2
                            print("detectados2: ", cont)
                            #print(f[a])
                                #print(a)
                            #print(f)
                                #print(i,"-",f[i])

                            #plt.plot(Sxx)
                            #plt.show()
                            #plt.pcolormesh(t, f, Sxx)
                            #plt.ylabel('Frequency [Hz]')
                            #plt.xlabel('Time [sec]')
                            #plt.show()
                            #plt.show()
                            #plt.plot(librosa.feature.zero_crossing_rate(x))
                            #plt.show()
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
                        data2 = archivo2x.readframes(-1)
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
                        a=frec-10
                        b=frec+10
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
                    dur = int(dur * 44100)
                    print(dur)
                    print("ingrese intensidad minima  ej:500")
                    inten = int(input())
                    cont_name = 0
                    b = pd.read_csv('nombres_individuales.csv')
                    name_csv = []
                    dif_csv = []
                    df1 = pd.DataFrame([], columns=['nombres', 'distancias'])
                    for i in range(len(valores)):
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
                        a = []
                        aux = 0

                        if (np.amax(signal) > inten):
                            for x in range(0, len(signal) - dur*3):
                                if signal[x] > inten:
                                    if ((len(a) == 0)):
                                        a.append(x)
                                    else:
                                        if (x - (a[aux]) > 15000 and (np.abs(signal[x + dur])) < (inten / 2)):
                                            a.append(x)
                                            aux = aux + 1
                        a2 = []
                        aux = 0
                        if (np.amax(signal2) > inten):
                            for z in range(0, len(signal2) - dur*3):
                                if signal2[z] > inten:
                                    if ((len(a2) == 0)):
                                        a2.append(z)

                                    else:
                                        if (z - (a2[aux]) > 15000 and (np.abs(signal2[z + dur])) < (inten / 2)):
                                            a2.append(z)
                                            aux = aux + 1
                        print("sonidos detectados en señal 1:", len(a))
                        print("sonidos detectados en señal 2:", len(a2))

                        print("Numero de muestra señal 1: ", a)
                        print("Tiempo de inicio del sonido detectado: ", np.round([n / 44100 for n in a], decimals=2))

                        print("posiciones señal 2: ", a2)
                        print("Tiempo de inicio del sonido detectado: ", np.round([n / 44100 for n in a2], decimals=2))

                        if (len(a) == len(a2) and len(a) != 0):
                            for valor in range(0, len(a)):
                                tiempo = a[valor] / 44.1
                                tiempo2 = a2[valor] / 44.1
                                t1 = tiempo - 5
                                t2 = tiempo + 200
                                t3 = tiempo2 - 5
                                t4 = tiempo2 + 200
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
                        #plt.subplot(211)
                        plt.title(str(valores[i]))
                        #plt.plot(signal)
                        #plt.plot(signal2)
                        #axes = plt.gca()
                        #axes.set_ylim([-700, 700])
                        #plt.subplot(212)
                        plt.title(str(valores[i+1]))
                        #plt.plot(signal2)
                        #plt.plot(signal)
                        #axes = plt.gca()
                        #axes.set_ylim([-700, 700])
                        #plt.show()
                        print(len(signal))
                        print(len(signal2))
                        f, t, Sxx = sg.spectrogram(signal, fs, nperseg=441)
                        plt.pcolormesh(t, f, Sxx)
                        axes = plt.gca()
                        axes.set_ylim([200, 2050])
                        plt.ylabel('Frecuencia [Hz]')
                        plt.xlabel('Tiempo [sec]')
                        plt.show()
                        plt.plot(signal)
                        plt.show()

                if option==8:
                    df = pd.read_csv('fin.csv')
                    a = (df.loc[:, '0']).astype('int')
                    b = (df.loc[:, '1']).astype('int')
                    c = (df.loc[:, '2']).astype('int')
                    d = (df.loc[:, '3']).astype('int')
                    aux=0

                    for i in range(0, len(df),19):
                        a1 = 0
                        b1 = 0
                        c1 = 0
                        d1 = 0
                        for j in range(0,19):
                            a1=a1+a[i+j]
                            b1=b1+b[i+j]
                            c1=c1+c[i+j]
                            d1=d1+d[i+j]
                        if(a1>b1 and a1>c1 and a1>d1):
                            print('1')
                        if(b1 > a1 and b1 > c1 and b1 > d1):
                            print('2')
                        if(c1>a1 and c1>b1 and c1>d1):
                            print('3')
                        if(d1>a1 and d1>b1 and d1>c1):
                            print('4')
                    #df1 = pd.DataFrame( valores_csv[:])
                    #df1.to_csv("nombres_valores.csv")

                if option == 82:
                    myaudio = AudioSegment.from_wav("1.wav")
                    myaudio2 = AudioSegment.from_wav("2.wav")
                    # silencio = silence.detect_nonsilent(myaudio, min_silence_len=1, silence_thresh=-32)

                    aux = True
                    contador = 16
                    comparar = [0, 2000]
                    while (aux == True):
                        silencio = silence.detect_silence(myaudio, min_silence_len=5, silence_thresh=contador)
                        if (not silencio) or (silencio == [comparar]) or (silencio[0][0] == 0) or (silencio[0][1] == 2000) or (len(silencio) > 10) :
                            if contador < -120:
                                print("error")
                                aux = False
                                print(contador)

                            contador = contador - 1
                        else:
                            # print(silencio)
                            # print(sonido)
                            aux = False

                    aux = True
                    contador = 16
                    comparar = [0, 2000]
                    while (aux == True):
                        silencio2 = silence.detect_silence(myaudio2, min_silence_len=5, silence_thresh=contador)
                        if (not silencio) or (silencio2 == [comparar]) or (silencio2[0][0] == 0) or (silencio2[0][1] == 2000) or (len(silencio2) > 10) or (silencio2[0][1] == 1999):
                            if contador < -120:
                                print("error")
                                aux = False
                                print(contador)
                            contador = contador - 1
                        else:
                            aux = False

                    if (len(silencio) == 0 or len(silencio2) == 0):
                        print("grabar denuevo")
                    else:

                        if diferencia < 0:
                            dif_iz = 0
                            dif_der = (diferencia * (-1))
                        if diferencia > 0:
                            dif_iz = diferencia
                            dif_der = 0
                        if diferencia == 0:
                            dif_iz = 0
                            dif_der = 0

                        sil_1 = ((silencio[0][1] / 1000) + dif_der)
                        sil_2 = ((silencio2[0][1] / 1000) + dif_iz)
                        print(dif_iz)
                        print(dif_der)
                        print("   ")
                        print(sil_1)
                        print(sil_2)
                        print(silencio)
                        print(silencio2)

                        if sil_1 < sil_2:
                            print("derecha")
                            print(sil_1 - sil_2)

                        if sil_1 > sil_2:
                            print("izquierda")
                            print(sil_1 - sil_2)
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
                    a = pd.read_csv('nombres_individuales.csv')
                    valores = a['nombres']
                    valores_csv = []
                    distancia =a['distancias']
                    #print(distancia)
                    #print(len(valores))
                    for i in range(0, int(len(valores)),2):
                        dis=int(distancia[i])
                        string= int(distancia[i])
                        array = np.zeros(50,dtype=int)
                        array[int(string/10)]=1


                        uno = str(str(valores[i])+".wav")
                        dos = str(str(valores[i+1]) + ".wav")
                        archivo = wave.open(uno, 'rb')
                        archivo2 = wave.open(dos, 'rb')
                        myaudio1 = AudioSegment.from_wav(uno)
                        myaudio2 = AudioSegment.from_wav(dos)
                        canales = archivo.getnchannels()
                        frames = archivo.getframerate()
                        fs = frames
                        datos = archivo.getparams()
                        samples = archivo.getsampwidth()
                        data = archivo.readframes(-1)
                        data2 = archivo2.readframes(-1)
                        signal = np.frombuffer(data, 'Int16')
                        signal2 = np.frombuffer(data2, 'Int16')

                        convolve = fftconvolve(signal, signal2, "same") / 10
                        convolve = np.round(convolve)
                        print(len(convolve))

                        if len(convolve)<9041:
                            np.append(convolve,[0])
                        prim = []
                        prim1 = []
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

                        rep=len(convolve)
                        for x in range(0, rep-19, 20):
                            #print(x)
                            prim.append(convolve[x])
                            prim2.append(convolve[x+1])
                            prim3.append(convolve[x+2])
                            prim4.append(convolve[x+3])
                            prim5.append(convolve[x+4])
                            prim6.append(convolve[x+5])
                            prim7.append(convolve[x+6])
                            prim8.append(convolve[x+7])
                            prim9.append(convolve[x+8])
                            prim10.append(convolve[x+9])
                            prim11.append(convolve[x+10])
                            prim12.append(convolve[x+11])
                            prim13.append(convolve[x+12])
                            prim14.append(convolve[x+13])
                            prim15.append(convolve[x+14])
                            prim16.append(convolve[x+15])
                            prim17.append(convolve[x+16])
                            prim18.append(convolve[x+17])
                            prim19.append(convolve[x+18])
                            prim20.append(convolve[x+19])

                            #print(x+18)

                        for z in range(0,50):
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
                        #print(i)
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


                    #print(len(valores_csv))

                    df1 = pd.DataFrame( valores_csv[:])
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



