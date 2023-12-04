import time
import cv2
import numpy as np
import random
import dlib
import sys

from funcs import *
from tastatura import Tastatura


sirina, visina = 480, 640
offset = (100, 80) #jedno dugme na tastaturi

eyeScale = 4.5 
scale = 0.3 
#kasnije koristimo za lepsi prikaz svih prozora
kamera = cv2.VideoCapture(0)
dim = (kamera.get(cv2.CAP_PROP_FRAME_HEIGHT), kamera.get(cv2.CAP_PROP_FRAME_WIDTH))

#dve glavne projekcije 
tastatura = CrnaSlika(dim)
kalibracija = CrnaSlika(dim)
keypoints = Tastatura(sirina  = sirina , visina = visina ,offset = offset )

detektor = dlib.get_frontal_face_detector()
prediktor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#postavlje tacke za odredivanje granica pogleda
tacke = [(offset),
           (sirina+offset[0], sirina-offset[0]),
           (sirina+offset[0], offset[1]),
           (offset[0],sirina-offset[0])]
cutFrame = []
num  = 0

#citamo koordinate ivica kako bismo omogucili pracenje pogleda i mapiranje
while(num<4): 

    ret, frame = kamera.read()   
    frame = cv2.flip(frame, 1)

    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    cv2.putText(kalibracija, 'Trepnite dok gledate tacku', tuple((np.array(dim)/3).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 3)
    cv2.circle(kalibracija, tacke[num], 20, (0, 255, 0), -1)

    faces = detektor(grayScale)
    if len(faces)> 1:
        print('višak lica')
        sys.exit()

    #detekcija lica i kljucnih tacki    
    for face in faces:

        PrikaziLice(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))
        landmarks = prediktor(grayScale, face) 
        PrikaziLandMarks(frame, landmarks, [0, 68], 'red') 

        #detekcija oka
        #tacke od interesa su unapred definisane
        koordinateOka = DesnoOko(landmarks, [42, 43, 44, 45, 46, 47])
        PrikaziOko(frame, koordinateOka, 'green')

    #detekcija zenice kao centralne figure kada izvucemo oko iz frame-a
    zenice = np.mean([koordinateOka[2], koordinateOka[3]], axis = 0).astype('int')

    #detekcija klika
    if Klik(koordinateOka):

        cutFrame.append(zenice)
        cv2.putText(kalibracija, 'ok',
                    tuple(np.array(tacke[num])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
        #nas "klik" odnosno da zazmurimo na jedno oko traje duze nego kada normalno trepnemo, tako da ako uspavamo program
        #mozemo preskociti slucajna zmurenja tj. normalno treptanje
        time.sleep(0.3)
        num = num + 1

    print(cutFrame, '    len: ', len(cutFrame))
    Prozor('projection', kalibracija)
    Prozor('frame', cv2.resize(frame,  (640, 360)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

#kalibracija
#nalazimo granicne vrednosti koji smo dobili ciljanjem prethodnih tacaka
#kao i offset za pogled
minX, maxX, minY, maxY = Kalibracija(cutFrame)
minY = minY - 40 #ovde je -40 da bi imali malo prostor oko oka u slucaju pomeranja glave
minX = minX - 40
kalibracijaOffset = [ minX, minY ]

cv2.putText(kalibracija, 'Kalibracija gotova.',
            tuple([int(dim[0]/3-10),int(dim[1]/3)+20]), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255), 2)
Prozor('kalibracija', kalibracija)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('tastatura')

pritisnuto = True
# prikaz pritisnutih tastera
text = "text: "
while(True):

    ret, frame = kamera.read()   
    frame = cv2.flip(frame, 1)  
    noviFrame = np.copy(frame[minY:maxY, minX:maxX, :])

    tastatura = CrnaSlika(dim)
    PrikaziTastaturu(frame = tastatura, keys = keypoints)
    textPrikaz = BelaSlika((200, 800))

    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detektor(grayScale) 
    if len(faces)> 1:
        print('visak lica...')
        sys.exit()

    for face in faces:
        PrikaziLice(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        landmarks = prediktor(grayScale, face)
        PrikaziLandMarks(frame, landmarks, [0, 68], 'red')

        koordinateOka = DesnoOko(landmarks, [42, 43, 44, 45, 46, 47])
        PrikaziOko(frame, koordinateOka, 'green')

    zenice = np.mean([koordinateOka[2], koordinateOka[3]], axis = 0).astype('int')

    #treba da redefinisemo koordinate zenica
    #za pomeraj koji smo odredili u kalibraciji
    koordZenice = np.array([zenice[0] - kalibracijaOffset[0], zenice[1] - kalibracijaOffset[1]])
    cv2.circle(noviFrame, (koordZenice[0], koordZenice[1]), int(Radius(koordinateOka)/1.5), (255, 0, 0), 3)

    #provera da li nije doslo do greske
    if ProveriZenice(koordZenice, noviFrame):

        #mapiranje pogleda
        pointer = Mapiraj(source = noviFrame[:,:, 0],dest = tastatura[:,:, 0], point = koordZenice)

        #sluzi kao pointer misa
        cv2.circle(tastatura, (pointer[0], pointer[1]), 10, (0, 255, 0), 3)

        if Klik(koordinateOka):

            pritisnuto = ProveriBrojku(keypoints = keypoints, koordX = pointer[1], koordY= pointer[0])
            if pritisnuto:
                text = text + pritisnuto

            time.sleep(0.6)

    cv2.putText(textPrikaz, text,(20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)

    # prozori
    Prozor('projection', tastatura)
    Prozor('frame', cv2.resize(frame, (int(frame.shape[1] *scale), int(frame.shape[0] *scale))))
    Prozor('noviFrame', cv2.resize(noviFrame, (int(noviFrame.shape[1] *eyeScale), int(noviFrame.shape[0] *eyeScale))))
    Prozor('textPrikaz', textPrikaz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# -------------------------------------------------------------------

kamera.release()
cv2.destroyAllWindows()