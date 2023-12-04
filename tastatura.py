import cv2
import numpy as np

def Tastatura(sirina , visina, offset):
    
    box = int(sirina / 3) 
    kolona = np.empty(10)
    kolona[0] = offset[0]
    for i in range(1,10):
        kolona[i] = kolona[i-1] + box

    vrsta = np.empty(3)
    vrsta[0] = offset[1]
    for i in range(1,3):
        vrsta[i] = vrsta[i-1] + box
    color_board = (250, 0, 100)

    keyPoints = []

    keyPoints.append(['1', (kolona[0], vrsta[0]), (kolona[0]-box/2, vrsta[0]-box/2), (kolona[0]+box/2, vrsta[0]+box/2)])
    keyPoints.append(['2', (kolona[1], vrsta[0]), (kolona[1]-box/2, vrsta[0]-box/2), (kolona[1]+box/2, vrsta[0]+box/2)])
    keyPoints.append(['3', (kolona[2], vrsta[0]), (kolona[2]-box/2, vrsta[0]-box/2), (kolona[2]+box/2, vrsta[0]+box/2)])
   
    keyPoints.append(['4', (kolona[0], vrsta[1]), (kolona[0]-box/2, vrsta[1]-box/2), (kolona[0]+box/2, vrsta[1]+box/2)])
    keyPoints.append(['5', (kolona[1], vrsta[1]), (kolona[1]-box/2, vrsta[1]-box/2), (kolona[1]+box/2, vrsta[1]+box/2)])
    keyPoints.append(['6', (kolona[2], vrsta[1]), (kolona[2]-box/2, vrsta[1]-box/2), (kolona[2]+box/2, vrsta[1]+box/2)])
   
    keyPoints.append(['7', (kolona[0], vrsta[2]), (kolona[0]-box/2, vrsta[2]-box/2), (kolona[0]+box/2, vrsta[2]+box/2)])
    keyPoints.append(['8', (kolona[1], vrsta[2]), (kolona[1]-box/2, vrsta[2]-box/2), (kolona[1]+box/2, vrsta[2]+box/2)])
    keyPoints.append(['9', (kolona[2], vrsta[2]), (kolona[2]-box/2, vrsta[2]-box/2), (kolona[2]+box/2, vrsta[2]+box/2)])
    

    return keyPoints

