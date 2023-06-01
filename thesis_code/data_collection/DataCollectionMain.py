import WebcamModule as wM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import ThesisCarControll_v1 as mM
import cv2
from time import sleep


#maxThrottle = 0.25
#motor = mM.Motor(2, 3, 4, 17, 22, 27)

mM.move(0, 1)
record = 0

while True:
    joyVal = jsM.getJS()
    #print(joyVal)
    steering = joyVal['axis1']
    throttle = joyVal['axis3']
    throttle = throttle * (-1) # because joystick was behaing opposite
    #print(steering)
    #throttle = joyVal['o']*maxThrottle
    if joyVal['share'] == 1:
        if record ==0: print('Recording Started ...')
        record +=1
        sleep(0.300)
    if record == 1:
        img = wM.getImg(True,size=[240,120])
        if throttle > 0.1 :
            dcM.saveData(img,steering)
    elif record == 2:
        mM.move(0,0)
        dcM.saveLog()
        record = 0

    mM.move(throttle,steering)
    cv2.waitKey(1)