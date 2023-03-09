#basic libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

#kivy package for map
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.lang import Builder

#Builder.load_file('car.kv')

#Lets import the AI from the agent file
from agent import Dqn

#adding this line
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

#initializing the last_x and last_y i.e, the cordinates
#used to keep the point in memory when we draw the sand in the map
last_x = 0
last_y = 0
n_points = 0
length = 0

#Model of Car or NN
#Feeding some input:
    #5 states:
    #3 actions: go left, go right, go straight
    #gamma: paramter to control the exploration and exploitation
states_i = 5
actions_i = 3
gamma_i = .9
model = Dqn(states_i,actions_i,gamma_i) 
#Lets suppose an example:
#These values are rotation values
#index 0 i.e, 0 degree corresponds to action go straight, 
#index 1 i.e, 20 degree corresponds to action rotate 20 degree which will go 20 degree right
#index 2 i.e, -20 degree corresponds to action rotate -20 degree which will go -20 degree left
action2rotation = [0, 20, -20]

# We will penalize the car if it goes to sand else +ve reward
#For each car will accumulate some rewards
last_reward = 0
#we will append the reward so that later we can use to to plot something for a episode
scores = []

#Initialize the map
#Sand will be pixels in the map i.e, an array 1 if sand else if no sand then 0
#At beginning there will be no sand so values will be zero
#Goal X is destination or say goal the car want to accomplish, 
#suppose there is some track and goal is to reach the tp left corner so car should go that way
#Goal Y is like back to home, once Goal X is achived then Goal Y will be triggred
# We will make the road as complicated as possible to make sure it reach both goals
# we have to avoid walls and the grass/sand
first_update = True
def init():
    global sand
    global x_destination
    global y_home
    global first_update
    sand = np.zeros((long_x, larg_y))
    x_destination = 20
    y_home = larg_y-20
    first_update = False
    
#initialize the last distance
#it gives the current distance of the car so 0 for now
last_distance = 0


#Create the CAR
#So to build it we need to understand some ethics how it works and what all info we 
#need to be aware of for example angle of car, velocity of car, rotation of car, sensors etc
#So there will be 3 sensor in our car, Sensor 1 will be check is there any object infront of car
#Sensor 2, any object on left
#Sensor 3, any object on right and then from these sensors we will recieve the signal
#Signal 1 is the signal recieved from Sensor 1
#Signal 2 is the signal recieved from Sensor 2
#Signal 3 is the signal recieved from Sensor 3
#this is calculated using density function
#signal 1 is the density of sand around sensor 1: we take squares of each of sensor
# which is 200x200, and for each of the squares we divide number of ones in the square by
#all number of cell in square, which 20x20 = 400 that gives density, because the ones corresponds to sand
#we do it for all sensor 
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor_x1 = NumericProperty(0)
    sensor_y1 = NumericProperty(0)
    sensor_1 = ReferenceListProperty(sensor_x1, sensor_y1)
    sensor_x2 = NumericProperty(0)
    sensor_y2 = NumericProperty(0)
    sensor_2 = ReferenceListProperty(sensor_x2, sensor_y2)
    sensor_x3 = NumericProperty(0)
    sensor_y3 = NumericProperty(0)
    sensor_3 = ReferenceListProperty(sensor_x3, sensor_y3)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        #allow to go left, right, straight
        ##pos is last position, position will be updated in the direction of velocity vector
        self.pos = Vector(*self.velocity) + self.pos 
        #how we gonna rotate the car going to left or right
        self.rotation = rotation
        #angle between x axis and the axis of the direction of the car
        self.angle = self.angle + self.rotation
        #once the car is moved then we have to update the sensor and the signal
        #so if car is rotated sensor will also get rotates, so we updated using rotate fn. and to which we add new pos
        #30 is the difference between sensor and car
        self.sensor_1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor_2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor_3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        #once sensor updated its time for signal
        #here we get x1 sensor and we take all the cell values from +10 to -10 then we do same for y cordinates from -10 to +10
        #we get square of 20x20 pixels sorrounding the sensor, and inside the square we sum all the ones, so 20x20 is 400 cells so thats
        # we divivded it by 400 to get the density of ones inside the square, thats to detect sand
        self.signal1 = int(np.sum(sand[int(self.sensor_x1)-10:int(self.sensor_x1)+10, int(self.sensor_y1)-10:int(self.sensor_y1)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor_x2)-10:int(self.sensor_x2)+10, int(self.sensor_y2)-10:int(self.sensor_y2)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor_x3)-10:int(self.sensor_x3)+10, int(self.sensor_y3)-10:int(self.sensor_y3)+10]))/400.
        #this below line *is for rewarding bad if it reaches the one of the edges in map
        ##########right edge################left edge#################top edge##############bottom edge############
        if self.sensor_x1>long_x-10 or self.sensor_x1<10 or self.sensor_y1>larg_y-10 or self.sensor_y1<10:
            #it will stop the car, worst value it can get is 1 i.e, bad reward
            self.signal1 = 1
        if self.sensor_x2>long_x-10 or self.sensor_x2<10 or self.sensor_y2>larg_y-10 or self.sensor_y2<10:
            self.signal2 = 1
        if self.sensor_x3>long_x-10 or self.sensor_x3<10 or self.sensor_y3>larg_y-10 or self.sensor_y3<10:
            self.signal3 = 1


class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
    
    def update(self, dt):
        global model
        global last_reward
        global scores
        global last_distance
        global x_destination
        global y_home
        global long_x
        global larg_y

        long_x = self.width
        larg_y = self.height
        if first_update:
            init()
        xx = x_destination - self.car.x
        yy = y_home - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180
        #creating last signal, since its obvious we have to take from all 3
        #adding orientation wrt goal beucase: if its heading towards goal then orientation willl be 0,
        # if it goes slightly towards right then orientation will be close to 45 degree
        # or left then -45 degree, adding -orientation in the car means stablizing the exploration i.e, both direction not just one
        #these 5 input will go to the agent as encoded vector
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        #and it return actions
        action = model.update(last_reward, last_signal)
        #update mean score
        scores.append(model.score())
        #we update rotation based on action
        rotation = action2rotation[action]
        #move the car based on rotation
        self.car.move(rotation)
        #we update the distance
        distance = np.sqrt((self.car.x - x_destination)**2 + (self.car.y - y_home)**2)
        #position updation
        self.ball1.pos = self.car.sensor_1
        self.ball2.pos = self.car.sensor_2
        self.ball3.pos = self.car.sensor_3

        #if car in sand, reduce the velocity also get bad reward
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            #if closer to goal get good reward, if it deviates from goal get slight bad reward
            #6 because it should keep usual speed
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        #last condition for rewards
        #if car comes to close to edges it get -1 reward
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x>self.width-10:
            self.car.x = self.width-10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height-10:
            self.car.y = self.height-10
            last_reward = -1
        #once reached goal
        #we update the x cordinate of goal as well as y cordinates
        #and then we update distance from car to goal
        if distance < 100:
            x_destination = self.width-x_destination
            y_home = self.height-y_home
        last_distance = distance

#Adding the paint tools
class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10
            touch.ud['line'] = Line(points = (touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x-last_x)**2 + (y-last_y)**2, 2))
            n_points += 1
            density = n_points/(length)
            touch.ud['line'].width = int(20*density+1)
            sand[int(touch.x) - 10 : int(touch.x)+10, int(touch.y)-10 : int(touch.y)+10]
            last_x = x
            last_y = y
#adding clear, save and load button
class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos = (parent.width, 0))
        loadbtn = Button(text='load', pos = (2*parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((long_x, larg_y))
    #to save the model and re use it later
    def save(self, obj):
        print('saving model')
        model.save()
        plt.plot(scores)
        plt.show()
    def load(self, obj):
        print('loading the last save model')
        model.load()
if __name__ == "__main__":
    CarApp().run()

#pip install spyder‑kernels==2.1.*