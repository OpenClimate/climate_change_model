""" This module contains the functions for iterating the social and climate 
coupled model (SoCCo)"""

import numpy as np
import pandas as pd
from scipy import stats

from . import social as sl
from . import climate as cl
    
#### randomUniformF ############################################################

def randomUniformF(nSamples=1):
    """
    returns random variates on scale (0,1).
    This function can replace perceivedBehavioralControlF()
    and efficacyF().
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

#### randomNormalF ############################################################
  
def randomNormalF(mean, sd, nSamples=1):
    """
    returns normal random variates.
    This function is used to initialize per capita emissions.
    pcEmissions = per capita emissions of CO2; Current values estimated
    by annualGHGemissionsInit/WorldPopnInit resulting if value of 5.049
    per person
    """  
    
    return np.random.normal(loc=mean,scale=sd,size=nSamples)


#### eIncrement ################################################################

def eIncrement(att, pbc, psn):
    """
    eIncrement[att_,pbc_,psn_]: rescales att and psn to -Inf to Inf and 
    then multiplies by pbc (0 to 1) to result in a increment in per 
    capita emissions
    att = attitude, pbc = perceivedBehavioralControl, psn = perceivedSocialNorm
    """
    attInv = stats.norm(loc=0.0,scale=1.0).ppf(att) # InverseCDF
    attInv[attInv==-np.inf]= min(10*attInv[attInv!=-np.inf].min(),-10) # avoid -inf 
    attInv[attInv==np.inf]= max(10*attInv[attInv!=np.inf].max(),10) # avoid +inf
    
    psnInv = stats.norm(loc=0.0,scale=1.0).ppf(psn) # InverseCDF
    psnInv[psnInv==-np.inf]= min(10*psnInv[psnInv!=-np.inf].min(),-10) # avoid -inf
    psnInv[psnInv==np.inf]= max(10*psnInv[psnInv!=np.inf].max(),10) # avoid +inf
    
    eDelIncrement = -(attInv + psnInv)*pbc
  
    return eDelIncrement
    
    
#### updatePCEmissions ################################################################

def updatePCEmissions(pcE, eff, pbc, tData,percepWindowSize,riskSens=1.0):
    """
    updatePCEmissions calculates a del pcE and then adds this to current pcE to
    return new pcE in 
    """
    # climateDat=testClimateData; yearCurrent=0; percepWindowSize=3; riskSens=1.0
    psn= sl.perceivedSocialNorm(pcE) #
    risk = sl.perceivedRisk(percepWindowSize, tData, riskSens)
    att = sl.attitude(risk, eff)
    # emissionsPC_Index = pcEmissionsToIndex(emissionsPC, mean=emissionsPC.mean(), sd=emissionsPC.std())
    pcE_Del = eIncrement(att, pbc, psn)
    pcE_New = pcE_Del + pcE
  
    return pcE_New
    

#### iterateOneStep ############################################################


def iterateOneStep(pcE_ts, tData_ts, co2_ts, eff, pbc, popN,percepWindowSize=3,riskSens=1.0):
    """
    Updates atm CO2, temperature and per capita emissions for one step (one year).
    """
    pcE_updated=updatePCEmissions(pcE_ts[:,-1], eff, pbc,tData_ts,percepWindowSize,riskSens)
    pcE_updated=np.atleast_2d(pcE_updated).transpose()
    pcE_vector=np.concatenate((pcE_ts, pcE_updated),axis=1)
    co2Del_ppm=cl.perCapitaEmissionsToDelPPM(pcE_updated, popN)
    co2_updated = np.array([co2Del_ppm + co2_ts[-1]])  # adds to last element of co2Current
    co2_vector = np.concatenate( [co2_ts, co2_updated] )
    rf = cl.computeRF(co2_vector)
    tDel=cl.compute_deltaT(rf)
    t_updated = np.array([tDel[-1] + tData_ts[-1]])  # adds to last element of co2Current
    t_vector = np.concatenate( [tData_ts, t_updated] )
        
    return pcE_vector,t_vector,co2_vector
    
    
#### iterateOneStep ############################################################

def iterateNsteps(pcE_init,tData_init, co2_init, nSteps, eff, pbc,popN,
            percepWindowSize=3,riskSens=1.0):
    """
    'Nsteps' updates of per capita emissions, temperature, and atm CO2 with each
    step being 1 year
    """
    
    for i in range(nSteps):
            
            pcE_init,tData_init,co2_init = iterateOneStep(pcE_init,tData_init, co2_init,
                eff, pbc,popN,percepWindowSize=3,riskSens=1.0)
                
    return pcE_init,tData_init,co2_init





