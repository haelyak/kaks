import random
from Org import *

def procreate(orgL,popSize,mutProb):
    '''Given a list of organisms orgL, obtain the next generation, popSize
large, by sampling with replacement.'''
    newPopL=[]
    for i in range(popSize):
        tempOrg=random.choice(orgL)
        newPopL.append(tempOrg.replicate(mutProb))
    return newPopL
        
def evoSim(popL,popSize,numGen,mutProb):
    '''Given a starting population popSize large, carry out a population
genetic simulation for numGen generations with mutProb probability of
mutation per replication.'''
    simL=[popL]
    for i in range(numGen):
        popL=procreate(popL,popSize,mutProb)
        simL.append(popL)
    return simL

def evoSimSelect(popL,popSize,numGen,mutProb,fitnessD):
    '''Given a starting population popSize large, carry out a population genetic
simulation with selection for numGen generations. The probability of
mutation per replication is mutProb, and fitnesses are specified by fitnessD.'''
    simL=[popL]
    for i in range(numGen):
        survivorL=cullPop(popL,fitnessD)
        popL=procreate(survivorL,popSize,mutProb)
        simL.append(popL)
    return simL
