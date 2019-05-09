from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse
from faceDetect import FaceDetect
from faceRecognition import FaceRecogntion


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=3,
    help='drone mesafesini değiştirmek için -d''yi kullan. 0-6 arası')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
    help='x eksenine bağlı uçuş değerini değiştirmek için -sx kullan. 0-480 arası')
parser.add_argument('-sy', '--saftey_y', type=int, default=55, 
    help='y eksenine bağlı uçuş değerlerini değiştirmek için -sy kullan. 0-360 arası')
parser.add_argument('-os', '--override_speed', type=int, default=2,
    help='hızı değiştirmek için -os kullan. 0-3 arası')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='oturumu sessions klasöründe bir görüntü sırası olarak kaydetmek için -ss kullan')
parser.add_argument('-D', "--debug", action='store_true',
    help='debug modunu etkinleştirmek için -D kullan. Her şey aynı şekilde çalışır, ancak drone''a hiçbir komut gönderilmez.')

args = parser.parse_args()

# drone hızı
S = 20

# face sizes
faceSizes = [1026, 684, 456, 304, 202, 136, 90]

# hızlanma modunda başlayan değerler
#########NOT: henüz test edilmedi
acc = [500,250,250,150,110,70,50]

# FPS
FPS = 25
# pencere boyutu
dimensions = (960, 720)

# uygun mevcut dizinde olup olmadığımızın kontrolü
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

class Drone(object):
    
    def __init__(self):
        self.tello = Tello()
        self.faceRec = FaceRecogntion()
        self.faceDet = FaceDetect()
        
        # Drone hızı 
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        # Kontroller
        self.send_rc_control = False
        self.OVERRIDE = False
        self.oSpeed = args.override_speed
        self.tDistance = args.distance
        
        

    def run(self):
        
        if not self.tello.connect():
            print("Bağlantı Başarısız")
            return

        if not self.tello.set_speed(self.speed):
            print("Hız en düşük seviye olarak yarlanmadı")
            return

        # eğer video akışı açıksa
        if not self.tello.streamoff():
            print("Video akışı durdulmadı")
            return

        if not self.tello.streamon():
            print("Video akışı başlatılamadı")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        self.tello.get_battery()
        
        # Traking modda yerinde dönüş ayarı
        szX = args.saftey_x
        # Traking modda yükseklik ayarı
        szY = args.saftey_y
        
        if args.debug:
            print("DEBUG MODE AKTİF!")

        while not should_stop:
            self.update()

            if frame_read.stopped:
                frame_read.stop()
                break

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame
            time.sleep(1 / FPS)

            # listen for key presses
            k = cv2.waitKey(20)

            # control fonksiyonu
            self.control(k)

            # Quit the software
            if k == 27:
                should_stop = True
                break
            
            if not self.OVERRIDE:
                
                # detect face
                frameRet, bboxes = self.faceDet.detectFaceOpenCVDnn(frameRet)
                
                # hedef boyutu
                tSize = faceSizes[self.tDistance]

                # merkez boyutlarımız
                cWidth = int(dimensions[0]/2)
                cHeight = int(dimensions[1]/2)

                noFaces = len(bboxes) == 0
                
                if self.send_rc_control and not self.OVERRIDE:
                    for (x, y, w, h) in bboxes:
  
                        # vektör hesaplaması için sınırlamalar
                        end_cord_x = w
                        end_cord_y = h
                        end_size =(w - x) * 2
                        
                        # hedef coordinatları
                        targ_cord_x = int((x + end_cord_x)/2)
                        targ_cord_y = int((y + end_cord_y)/2)
                        
                        # yüz için vektör hesaplaması
                        vTrue = np.array((cWidth,cHeight,tSize))
                        vTarget = np.array((targ_cord_x,targ_cord_y,end_size))
                        vDistance = vTrue-vTarget

                        
                        if not args.debug:
                            # turning
                            if vDistance[0] < -szX:
                                self.yaw_velocity = S
                                
                            elif vDistance[0] > szX:
                                self.yaw_velocity = -S  

                            else:
                                self.yaw_velocity = 0
                            
                            # up & down
                            if vDistance[1] > szY: 
                                self.up_down_velocity = S    
                            elif vDistance[1] < -szY:
                                self.up_down_velocity = -S
                            else:
                                self.up_down_velocity = 0
                            """
                            F = 0
                            if abs(vDistance[2]) > acc[self.tDistance]:
                                F = S
                            """
                            # forward & backward
                            if vDistance[2] > 10:
                                self.for_back_velocity = 10  # S + F
                            elif vDistance[2] < - 10:
                                self.for_back_velocity = -10  # S - F
                            else:
                                self.for_back_velocity = 0


                        # yeşil çember
                        cv2.circle(frameRet, (targ_cord_x, targ_cord_y), 10, (0,255,0), 2)

                        # Dronun vektör konumu
                        cv2.putText(frameRet, str(vDistance), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        

                    # Tanınan yüz yoksa 
                    if noFaces:
                        self.yaw_velocity = 0
                        self.up_down_velocity = 0
                        self.for_back_velocity = 0
                        print("HEDEF BULUNAMADI")
                        
            elif self.OVERRIDE:
                face_locations, face_names = self.faceRec.recognition(frameRet)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frameRet, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frameRet, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frameRet, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
            # Açılan pencerenin merkez noktası
            cv2.circle(frameRet, (cWidth, cHeight), 10, (0, 0, 255), 2)

            dCol = lerp(np.array((0, 0, 255)), np.array((255, 255, 255)), self.tDistance+1/7)

            if self.OVERRIDE:
                show = "OVERRIDE: {}".format(self.oSpeed)
                dCol = (255,255,255)
            else:
                show = "AI: {}".format(str(self.tDistance))

            # sol alt köşede OVERRIDE ya da AI bilgisi
            cv2.putText(frameRet,show,(32,664),cv2.FONT_HERSHEY_SIMPLEX,1,dCol,2)

            # Frame gösterimi
            cv2.imshow(f'Drone Tracking...', frameRet)
            
        
        self.tello.get_battery()
        cv2.destroyAllWindows()
        self.tello.end()

    def control(self,k):
        
        # Mesafeyi 0 ayarlamak için 
        if k == ord('0'):
            if not self.OVERRIDE:
                print("mesafe = 0")
                self.tDistance = 0

        # Mesafeyi 1 ayarlamak için
        if k == ord('1'):
            if self.OVERRIDE:
                self.oSpeed = 1
            else:
                print("mesafe = 1")
                self.tDistance = 1

        # Mesafeyi 2 ayarlamak için
        if k == ord('2'):
            if self.OVERRIDE:
                self.oSpeed = 2
            else:
                print("mesafe = 2")
                self.tDistance = 2
                    
        # Mesafeyi 3 ayarlamak için
        if k == ord('3'):
            if self.OVERRIDE:
                self.oSpeed = 3
            else:
                print("mesafe = 3")
                self.tDistance = 3
            
        # Mesafeyi 4 ayarlamak için
        if k == ord('4'):
            if not self.OVERRIDE:
                print("mesafe = 4")
                self.tDistance = 4
                    
        # Mesafeyi 5 ayarlamak için
        if k == ord('5'):
            if not self.OVERRIDE:
                print("mesafe = 5")
                self.tDistance = 5
                    
        # Mesafeyi 6 ayarlamak için
        if k == ord('6'):
            if not self.OVERRIDE:
                print("mesafe = 6")
                self.tDistance = 6

        # takeoff = t
        if k == ord('t'):
            if not args.debug:
                print("Kalkıyor !!!")
                #self.tello.takeoff()
                self.tello.get_battery()
            self.send_rc_control = True

        # land = l
        if k == ord('l'):
            if not args.debug:
                print("İniyor !!!")
                self.tello.land()
            self.send_rc_control = False

        # OVERRIDE control için 8
        if k == ord('8'):
            if not self.OVERRIDE:
                self.OVERRIDE = True
                print("Klavye kontrolü aktif")
                self.keyboardControl(k)
            else:
                self.OVERRIDE = False
                print("Klavye kontrolü devre dışı")
        if self.OVERRIDE:
            self.keyboardControl(k)
        
    def keyboardControl(self,k):
        if self.OVERRIDE:
            # s = backward & w = forward
            if k == ord('w'):
                self.for_back_velocity = int(S * self.oSpeed)
            elif k == ord('s'):
                self.for_back_velocity = -int(S * self.oSpeed)
            else:
                self.for_back_velocity = 0

            # a = left & b = rigth (dönüş)
            if k == ord('d'):
                self.yaw_velocity = int(S * self.oSpeed)
            elif k == ord('a'):
                self.yaw_velocity = -int(S * self.oSpeed)
            else:
                self.yaw_velocity = 0
            # q = fly up & e = fly up
            if k == ord('e'):
                self.up_down_velocity = int(S * self.oSpeed)
            elif k == ord('q'):
                self.up_down_velocity = -int(S * self.oSpeed)
            else:
                self.up_down_velocity = 0

            # c = rigth & z = left
            if k == ord('c'):
                self.left_right_velocity = int(S * self.oSpeed)
            elif k == ord('z'):
                self.left_right_velocity = -int(S * self.oSpeed)
            else:
                self.left_right_velocity = 0

        
    def battery(self):
        return self.tello.get_battery()[:2]
    

    def update(self):
        
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

def lerp(a, b, c):

    return a + c*(b-a)

def main():
    
    drone = Drone()
    drone.run()


if __name__ == '__main__':
    main()
