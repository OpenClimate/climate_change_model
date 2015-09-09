"""This module contains the social functions for the coupled social
and climate model SoCCo"""

import numpy as np
import pandas as pd
from scipy import stats

from . import climate as cl

#### popIntoNgroups ############################################################

def popIntoNgroups(popTotal,nGroups,beta_a=1,beta_b=1):
    """Distributes total population popTotal into n groups that may 
    be equal or uequal"""
    
    popFrac=stats.beta.rvs(a=beta_a, b=beta_b, size=nGroups)
    popFrac=popFrac/popFrac.sum()
    popN=popFrac*popTotal 
    
    return popN
    

#### perceivedBehavioralControlF ###############################################

def perceivedBehavioralControlF(nSamples=1):
    """
    returns efficacy on scale (0,1) 
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

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
    xDiff_Scaled=stats.uniform(loc=-1.0,scale=2.0).cdf(xDiff) # range from loc to loc+scale
    
    return xDiff_Scaled


#### EfficacyF #################################################################

def efficacyF(nSamples=1):
    """
    returns efficacy on scale (0,1) 
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

#### perceivedRisk #######################################################################

def perceivedRisk(tLag,tData_ts,beta=1.0):
    """ returns perceived risk on scale (0,1)
    """
    myPerceivedRisk = beta*cl.climatePerturbationF(tLag, tData_ts)
    return stats.norm(loc=0,scale=1).cdf(myPerceivedRisk)
    

#### attitude ##################################################################

def attitude(perceivedRisk, eff):
    """A = attitude or behavioral intention, scaled
    on (0, 1), with two components: Risk Perception & Efficacy
    (utility).Role of efficacy only comes into play if the perceived risk
    is sufficiently high, i.e., >= 0. 1 corresponds to increase in per capita
    Emissions and 0 leads to reduction in per capita Emissions.
    """
    
    perceivedRiskInv=stats.norm(loc=0,scale=1).ppf(perceivedRisk) # inverseCDF
    myAttitudeInv = perceivedRisk*eff # larger value -> more motivation to reduce pcE
    myAttitude = 1 - stats.norm(loc=0,scale=1).cdf(myAttitudeInv) # reversing so 0 -> lower pcE, 1-> more pcE
    return myAttitude
    
    
