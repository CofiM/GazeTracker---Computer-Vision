import cv2
import numpy as np
import random
import time
import dlib
import sys

granica = 0.13 #odnos za ose oka

boje = {'green': (0,255,0),
              'blue':(255,0,0),
              'red': (0,0,255),
              'yellow': (0,255,255),
              'white': (255, 255, 255),
              'black': (0,0,0)}


def CrnaSlika(vel):
    str = (np.zeros((int(vel[0]), int(vel[1]), 3))).astype('uint8')
    return str

def BelaSlika(vel):
    str = (np.zeros((int(vel[0]), int(vel[1]), 3)) + 255).astype('uint8')
    return str

def Prozor(naslov, proz):
    cv2.namedWindow(naslov)
    cv2.imshow(naslov,proz)

def PrikaziLice(frame, temp, boja, vel):
    levo, gore, desno, dole = temp[0], temp[1], temp[2], temp[3]
    cv2.rectangle(frame, (levo-vel[0], gore-vel[1]), (desno+vel[0], dole+vel[1]),boje[boja], 5)

def polovina(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def DesnoOko(landmarks, tacke):
    levo = (landmarks.part(tacke[0]).x, landmarks.part(tacke[0]).y)
    desno = (landmarks.part(tacke[3]).x, landmarks.part(tacke[3]).y)
    gore = polovina(landmarks.part(tacke[1]), landmarks.part(tacke[2]))
    dole = polovina(landmarks.part(tacke[5]), landmarks.part(tacke[4]))
    return levo, desno, gore, dole

def PrikaziOko(frame, koord, boja):
    cv2.line(frame, koord[0], koord[1], boje[boja], 2)
    cv2.line(frame, koord[2], koord[3], boje[boja], 2)

def PrikaziLandMarks(frame, landmarks, tacke, boja):
    for temp in range(tacke[0], tacke[1]):
        x = landmarks.part(temp).x
        y = landmarks.part(temp).y
        cv2.circle(frame, (x, y), 4, boje[boja], 2)

def Klik(koord):
    klik = False
    xOsa = np.sqrt((koord[1][0]-koord[0][0])**2 + (koord[1][1]-koord[0][1])**2)
    yOsa = np.sqrt((koord[3][0]-koord[2][0])**2 + (koord[3][1]-koord[2][1])**2)
    odnos = yOsa / xOsa
    if odnos < granica: 
        klik = True
    return klik

def Kalibracija(frame):
    maxX = np.transpose(np.array(frame))[0].max()
    minX = np.transpose(np.array(frame))[0].min()
    maxY = np.transpose(np.array(frame))[1].max()
    minY = np.transpose(np.array(frame))[1].min()

    return minX, maxX, minY, maxY

def ProveriZenice(koordZenice, novFrame):
    check = False
    x = ((koordZenice[0] > 0) & (koordZenice[0] < novFrame.shape[1]))
    y = ((koordZenice[1] > 0) & (koordZenice[1] < novFrame.shape[0]))
    if x and y:
        check = True

    return check

def Mapiraj(source, dest, point):
    ratio = (np.array(dest.shape) / np.array(source.shape))  #.astype('int')
    point = (point * ratio).astype('int')
    return point

def PrikaziTastaturu(frame, keys):
    for key in keys:
        cv2.putText(frame, key[0], (int(key[1][0]),int(key[1][1])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 100), thickness = 3)
        cv2.rectangle(frame, (int(key[2][0]),int(key[2][1])), (int(key[3][0]),int(key[3][1])), (255, 250, 100), thickness = 4)

def ProveriBrojku(keypoints, koordX, koordY):
    check = False
    for key in range(0, len(keypoints)):
        uslov1 = np.mean(np.array([koordY, koordX]) > np.array(keypoints[key][2]))
        uslov2 = np.mean(np.array([koordY, koordX]) < np.array(keypoints[key][3]))
        if (int(uslov1 + uslov2) == 2):
            check = keypoints[key][0]
            break
    return check

def Radius(koord):
    radius = np.sqrt((koord[3][0]-koord[2][0])**2 + (koord[3][1]-koord[2][1])**2)
    return int(radius)
