# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

###################importing the brain, Dqn stands for deep Q learning ######
# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9) #Dqn(state, no. of possible action (forward, left, right), gamma factor)
action2rotation = [0,20,-20]  #[initial action, action 1 go right 20 degree, action 2 go left 20 degree]
last_reward = 0 #reward go up if not go into sand otherwise gets deduct
scores = []

# Initializing the map
first_update = True
def init():
    global sand #array of the pixel of the map, 1 if has sand, 0 if no sand. At beginning there's no sand so all 0 after we draw the line then will begin to have 1
    global goal_x  
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur)) #initialize all the arrays in sand with zero
    goal_x = 20 #destination at upper corner of the map
    goal_y = largeur - 20 #(goal_x, goal_y) = upper distance of the map
    first_update = False

# Initializing the last distance
last_distance = 0  #from current distance to the goal

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)  #angle between the x-axis and the direction of the car
    rotation = NumericProperty(0) #rotation mentioned above, either 0, 20 or -20
    velocity_x = NumericProperty(0) 
    velocity_y = NumericProperty(0) 
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)  #detecting if theres anything at the front of the car
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0) #detecting if theres anything at the left of the car
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0) #detecting if theres anything at the right of the car
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)  #density of the sense for sensor1 for each block of square (20x20 1s around each sensor), calcualted by number of 1/total no. of cell in each square
    signal2 = NumericProperty(0)  #density of the sense for sensor2
    signal3 = NumericProperty(0)  #density of the sense for sensor3

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos #new position of the car + last position of the car
        self.rotation = rotation #know to rotate to left or right
        self.angle = self.angle + self.rotation #angle b/t x-axis and the direction of the car
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos  #update the position. vector(distance b/t the car and sensor, distance b/t car and what car detects)
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. #take x & y coordicate and then take the range +/- 10 element in it. check how many 1s and divide by the total number square  
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1. #penalty,  signal = 1 is the most severe penalty. If the car goes a cross those 4 lines then most severe
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt): #neural network at the heart of the AI game

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] #last signal of the 3 sensor. with orientation of the car wrt the goal of the car, if the car is headinig towards the goal then it will be 0, if to the left will 45 degree or right then -45degree
          #second -orientation make sure the car explores in opposite direction
        action = brain.update(last_reward, last_signal) ######the output of our neural network. 
        scores.append(brain.score()) #update the score
        rotation = action2rotation[action]  #computer will select the action and get the location
        self.car.move(rotation)  #move the car according to the car 
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) #update the distance of the car
        self.ball1.pos = self.car.sensor1  #ball represent the sensor on the map
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)  #slow down if go to the sand (reduce velodity to 1 from 6)
            last_reward = -1 #reward between -1 and +1, -1 is the worst
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) #if it isn't on sand, keep usual speed
            last_reward = -0.1 # -0.2
            if distance < last_distance:
                last_reward = 0.1 #if closer to the goal then get positive award, otherwise negative award

        if self.car.x < 10: #if car get to close the left edge of the map
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10: #if car get to close the right edge of the map
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10: #if car get to close the bottom edge of the map
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10: #if car get to close the top edge of the map
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100: #update the goal when the goal is reach, changes it to the bottom right of the corner
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
