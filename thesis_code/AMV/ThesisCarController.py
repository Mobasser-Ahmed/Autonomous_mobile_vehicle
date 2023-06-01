from __future__ import division
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO
pwm = Adafruit_PCA9685.PCA9685()
 
servo_pin = 15 #  servo connect to PWM 15

CENTER= 436 #Steer servo car go forward
speedBase = 3000
GPIO.setmode(GPIO.BCM) # GPIO number  in BCM mode
GPIO.setwarnings(False)
#define L298N(Model-Pi motor drive board) GPIO pins
IN1 = 23  #right motor direction pin
IN2 = 24  #right motor direction pin
IN3 = 27  #left motor direction pin
IN4 = 22  #left motor direction pin
ENA = 0  #Right motor speed PCA9685 port 0
ENB = 1  #Left motor speed PCA9685 port 1
# Define motor control  pins as output
GPIO.setup(IN1, GPIO.OUT)   
GPIO.setup(IN2, GPIO.OUT) 
GPIO.setup(IN3, GPIO.OUT)   
GPIO.setup(IN4, GPIO.OUT) 
pwm.set_pwm_freq(60)

def changespeed(speed):
	speed = (speedBase* abs(speed) ) + (abs(speed) * 1000)
	speed = int (speed)
	pwm.set_pwm(ENA, 0, speed)
	pwm.set_pwm(ENB, 0, speed)

def stopcar():
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2, GPIO.LOW)
	GPIO.output(IN3, GPIO.LOW)
	GPIO.output(IN4, GPIO.LOW)
	changespeed(0)
	
def move(speed, steering):
    steering = steering * 100
    steering = int(steering)
    #print(steering + ' ' + speed)
    pwm.set_pwm(servo_pin, 0, CENTER + steering)
    if speed == 0 :
        stopcar()
    elif speed > 0 :    
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        changespeed(speed)
    elif speed < 0 :
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        changespeed(speed)
        

    


