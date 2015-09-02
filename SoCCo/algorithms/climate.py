### This folder contains the climate functions for the coupled social
### and climate model SoCCO

import numpy as np
import pandas as pd
# import scipy.stats as sp
from scipy import stats


#### computeRF #################################################################

def computeRF(co2c, initial=283.9660):
    """Radiative forcing function used in Forrest's climate impulse model.
    This is one of Forrest's functions, so information in compute_deltaT.
    """
    alpha = 5.35
    rf = alpha*np.log(co2c/initial)
    return rf

#### compute_deltaT ############################################################

def compute_deltaT(rf):
    """
    Compute temperature change from radiative forcing provided by computeRF. 
    This function implements Forrest Hoffman's climate impulse model (see his R 
    code in this folder).This function provides a sequence of delta T's given a
    sequence of atm CO2 concentrations.CO2 is CO2 mole fraction, which should be
    equivalent to ppm.  T in degrees Kelvin. The model is published at 
    http:// onlinelibrary.wiley.com/doi/10.1002/2013 JG002381/abstract
    with supplmentary material at
    http : // onlinelibrary.wiley.com/doi/10.1002/2013 JG002381/suppinfo
    
    """
    
    c1 = 0.631
    c2 = 0.429
    d1 = 8.4
    d2 = 409.5
    
    # Adjust c1 and d1 values to better correspond with CMIP5 results
    sf = 0.1054275
    c1 = c1 - (sf * c1)
    d1 = d1 - (sf * d1)
    
    #import ipdb ; ipdb.set_trace()
    int_big_delta_T = np.empty(len(rf))
    int_big_delta_T[0] = 0.

    for j in range(1, len(rf)): # returns 1 to 10 if len(time)=11
        # k = rev(0:j) - 1; j=1
        #print j
        k=np.arange(j+1)[::-1]
        delta_T = (c1/d1) * np.exp(-k/d1) + (c2/d2) * np.exp(-k/d2)
        
        if isinstance(rf, pd.DataFrame):     
            int_big_delta_T[j] = rf.iloc[0] * delta_T[0]
        else:
            int_big_delta_T[j] = rf[0] * delta_T[0]
            
        for i in np.arange(0,j)+1: # should go from 1 for j=1 to 1 to 10 for j=10
            # print i
            # i=0
            if isinstance(rf, pd.DataFrame):     
                int_big_delta_T[j] = int_big_delta_T[j] + rf.iloc[i] * delta_T[i]
            else:
                int_big_delta_T[j] = int_big_delta_T[j] + rf[i] * delta_T[i]
            
    return int_big_delta_T


#### perCapitaEmissionsToDelPPM ################################################

def perCapitaEmissionsToDelPPM(perCapCO2EmissionsN, popN,
    FracAbsorbed=0.5, GtCperPPM=2130000000.0):
    """Converts per capita CO2 emmissions to a change in atmospheric ppm"""
    
    co2EmissionsN = perCapCO2EmissionsN*popN
    co2EmissionsTotal=co2EmissionsN.sum() # sum across groups
    co2EmissionsToAtm = co2EmissionsTotal*(1 - FracAbsorbed) # removing absorbed
    cEmissionsToAtm = co2EmissionsToAtm/3.664 # Convert CO2 to C
    deltaPPM = cEmissionsToAtm /GtCperPPM;   
    
    return deltaPPM
    
 
 #### pcEmissionsToIndex ########################################################

def pcEmissionsToIndex(pcE, mean, sd):
    """
    Standardizes pcEmissions to a standard index using a normal cdf.
    with a given mean and sd.
    """  
    
    pcE_Scaled=sp.norm(loc=mean,scale=sd).cdf(pcE)

    return pcE_Scaled

#### pcIndexToEmissions ########################################################

def pcIndexToEmissions(pcE_Scaled, mean, sd):
    """
    Standardizes pcEmissions to a standard index using a normal cdf.
    The standardization can be done using a standard normal distribution (0,1)
    or using the observed mean and variance.  The standard normal distribution
    can be adjusted if needed.
    """  
    pcE=sp.norm(loc=mean,scale=sd).ppf(pcE_Scaled)

    return pcE
       
                
#### climatePerturbation #################################################################

def climatePerturbationF(windowWidth, tData_ts):
    """
    Takes last value in array and subtracts mean of previous windowWidth values
    """  
    if isinstance(tData_ts, pd.DataFrame):     
        perturbation=tData_ts.iloc[-1]-tData_ts.iloc[-2:(-2-windowWidth):-1].mean()
    else:
        perturbation=tData_ts[-1]-tData_ts[-2:(-2-windowWidth):-1].mean()
        
    return perturbation


def climatePerturbation_LeftisMoreRecent(currentIndex, windowWidth, data):
    """
    """  
    if isinstance(data, pd.DataFrame):     
        perturbation=data.iloc[currentIndex]-data.iloc[currentIndex:windowWidth].mean()
    else:
        perturbation=data[currentIndex]-data[currentIndex:windowWidth].mean()
        
    return perturbation

                   
                                                         
                      

                         
                                                  
                                                                                                    
                            
                               
                                     
