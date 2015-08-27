### This folder contains the climate and social functions for the coupled social
### and climate model

import numpy as np
import pandas as pd
import scipy.stats as sp

from . import climate as cl

    
#### popIntoNgroups ############################################################

def popIntoNgroups(popTotal,nGroups,beta_a=1,beta_b=1):
    """Distributes total population popTotal into n groups that may 
    be equal or uequal"""
    
    popFrac=sp.beta.rvs(a=beta_a, b=beta_b, size=nGroups)
    popFrac=popFrac/popFrac.sum()
    popN=popFrac*popTotal 
    
    return popN
    
if False: # Using function
    popTotal=7130010000 # Wolfram: QuantityMagnitude[CountryData["World", "Population"]]
    popN=popIntoNgroups(popTotal,nGroups=10)
    np.isclose(popTotal, popN.sum(), rtol=1e-05, atol=1e-08, equal_nan=False) # True


#### perceivedBehavioralControlF ###############################################

def perceivedBehavioralControlF(nSamples=1):
    """
    returns efficacy on scale (0,1) 
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

if False: # Using function
    perceivedBehavioralControlF(10)


#### perceivedSocialNorm ###############################################

def perceivedSocialNorm(xVect):
    """
    PSN = perceived social norm returns value scaled btwn[0, 1]. 
    Lower values (<0.5) mean that the per carbon emmissions will shift to 
    lower values while higer values (>0.5) mean the shift will be to 
    higher per capita emissions. PSNs constrain the actual changes in
    individual behavior due to perceived behaviors of others in the 
    population.
    """
    xDiff = xVect-xVect.mean()
    # xDiff_Scaled=sp.norm(loc=0,scale=1).cdf(xDiff)
    xDiff_Scaled=sp.uniform(loc=-1.0,scale=2.0).cdf(xDiff) # range from loc to loc+scale
    
    return xDiff_Scaled

if False: # Using function
    xVect=np.array([1,2,1,3])
    perceivedSocialNorm(xVect)

#### EfficacyF #################################################################

def efficacyF(nSamples=1):
    """
    returns efficacy on scale (0,1) 
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

if False: # Using function
    efficacyF(10)


#### perceivedRisk #######################################################################

def perceivedRisk(tLag,tData_ts,beta=1.0):
    """ returns perceived risk on scale (0,1)
    """
    myPerceivedRisk = beta*cl.climatePerturbationF(tLag, tData_ts)
    return sp.norm(loc=0,scale=1).cdf(myPerceivedRisk)
    
if False: # Using function
    testData=np.array([100,110,120,100])
    perceivedRisk(3,testData,beta=1.0)
    

#### attitude ##################################################################

def attitude(perceivedRisk, eff):
    """A = attitude or behavioral intention, scaled
    on (0, 1), with two components: Risk Perception & Efficacy
    (utility).Role of efficacy only comes into play if the perceived risk
    is sufficiently high, i.e., >= 0. 1 corresponds to increase in per capita
    Emissions and 0 leads to reduction in per capita Emissions.
    """
    
    perceivedRiskInv=sp.norm(loc=0,scale=1).ppf(perceivedRisk) # inverseCDF
    myAttitudeInv = perceivedRisk*eff # larger value -> more motivation to reduce pcE
    myAttitude = 1 - sp.norm(loc=0,scale=1).cdf(myAttitudeInv) # reversing so 0 -> lower pcE, 1-> more pcE
    return myAttitude
    
    
if False: # Using function
    testData=np.array([100,110,120,100])
    perRisk=perceivedRisk(0,3,testData,beta=1.0)
    eff=efficacyF(1)
    attitude(perRisk, eff)



    


