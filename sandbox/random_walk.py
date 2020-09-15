import matplotlib
matplotlib.use('TkAgg')

import pylab as PL
import random as RD
import scipy as SP

from PyCX_master import pycxsimulator

RD.seed()

populationSize = 1000
noiseLevel = 1

def init():
    global time, agents
    
    time = 0
    
    agents = []
    for i in range(populationSize):
        newAgent = [RD.gauss(0, 1), RD.gauss(0, 1)]
        agents.append(newAgent)
        
def draw():
    PL.cla()
    x = [ag[0] for ag in agents]
    y = [ag[1] for ag in agents]
    PL.plot(x, y, 'gh')
    PL.axis('scaled')
    PL.axis([-200, 200, -200, 200])
    PL.title('t = ' + str(time))
    
def step():
    global time, agents
    
    time += 1
    
    for ag in agents:
        ag[0] += RD.gauss(0, noiseLevel)
        ag[1] += RD.gauss(0, noiseLevel)

pycxsimulator.GUI().start(func=[init,draw,step])