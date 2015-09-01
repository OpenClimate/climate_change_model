### This folder contains the climate and social functions for the coupled social
### and climate model

import numpy as np
import pandas as pd
import scipy.stats as sp



#### computeRF #################################################################

def computeRF(co2c, initial=283.9660):
    """Radiative forcing function used in Forrest's climate impulse model.
    This is one of Forrest's functions, so information in compute_deltaT.
    """
    alpha = 5.35
    rf = alpha*np.log(co2c/initial)
    return rf


computeRFv = np.vectorize(computeRF) # Vectorized version that not actually used.

if False: # Using function
    co2 = np.linspace(290, 300, 11)
    rf = computeRF(co2)

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

if False: # Using function
    # time = range(2000,2011)
    
    ### Full time series
    co2 = np.linspace(290, 300, 11)
    rf = computeRF(co2)
    compute_deltaT(rf)
    
    ### Splitting full time series into two non overlapping sets: result si nto the same as above
    rf_1st = computeRF(co2[0:4])
    compute_deltaT(rf_1st)
    rf_2nd = computeRF(co2[4:])
    compute_deltaT(rf_2nd)
    
    len(time)
    len(co2)
    len(rf)
    

    ### comparison to R code results ### SAME ANSWER in R or Python! So code works!!!!!!

    head(model_rf)
    # 0.5714259 0.5871915 0.5995151 0.6131547 0.6223029 0.6327609
    tail(model_rf)
    # 1.661221 1.696176 1.720112 1.750476 1.786181 1.814206
    
    head(model_delta_T)
    # 0.00000000 0.08289799 0.11837983 0.15055578 0.17949723 0.20570794
    tail(model_delta_T)
    # 0.9424666 0.9607743 0.9788250 0.9971429 1.0161051 1.0350501
    
    
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
    
if False: # Using fnction
    pcE=randomNormalF(5.049, 0.5, 10)
    popTotal=7130010000 # Wolfram: QuantityMagnitude[CountryData["World", "Population"]]
    popN=popIntoNgroups(popTotal,nGroups=10)
    perCapitaEmissionsToDelPPM(pcE, popN)
 
 #### pcEmissionsToIndex ########################################################

def pcEmissionsToIndex(pcE, mean, sd):
    """
    Standardizes pcEmissions to a standard index using a normal cdf.
    with a given mean and sd.
    """  
    
    pcE_Scaled=sp.norm(loc=mean,scale=sd).cdf(pcE)

    return pcE_Scaled


if False: # Using function
    pcE=randomNormalF(5.049, 0.5, 10)
    pcEmissionsToIndex(pcE,mean=pcE.mean(),sd=pcE.std())
    pcEmissionsToIndex(pcE,mean=0.0,sd=1.0)
      
       
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


if False: # Using function
    pcE=randomNormalF(5.049, 0.5, 10)
    pce_scaled=pcEmissionsToIndex(pcE,mean=0.0,sd=1.0)
    pcIndexToEmissions(pce_scaled,mean=0.0,sd=1.0)          
             
                
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

if False: # Using function
    data=np.array([100,110,120,111])
    climatePerturbationF(3, data)
    perturbation=testData[1]-testData[-1:3].mean()
    
    
    
def climatePerturbation_LeftisMoreRecent(currentIndex, windowWidth, data):
    """
    """  
    if isinstance(data, pd.DataFrame):     
        perturbation=data.iloc[currentIndex]-data.iloc[currentIndex:windowWidth].mean()
    else:
        perturbation=data[currentIndex]-data[currentIndex:windowWidth].mean()
        
    return perturbation

                   
                                                         
                      

                         
                                                  
                                                                                                    
                            
                               
                                     
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


#### randomUniformF ############################################################

def randomUniformF(nSamples=1):
    """
    returns random variates on scale (0,1).
    This function can replace perceivedBehavioralControlF()
    and efficacyF().
    """  
    return np.random.uniform(low=0.0, high=1.0, size=nSamples)
    

if False: # Using function
    randomUniformF(10)

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


if False: # Using function
    randomNormalF(5.049, 0.5, 10)


#### Archived version
#   
#    def pcEmissionsToIndex(pcE,stdNorm=False):
#    """
#    Standardizes pcEmissions to a standard index using a normal cdf.
#    The standardization can be done using a standard normal distribution (0,1)
#    or using the observed mean and variance.  The standard normal distribution
#    can be adjusted if needed.
#    """  
#    
#    if stdNorm:
#        pcE_Scaled=sp.norm(loc=0,scale=1).cdf(pcE)
#    else:
#        mean=pcE.mean()
#        sd=pcE.std()
#        pcE_Scaled=sp.norm(loc=mean,scale=sd).cdf(pcE)
#
#    return pcE_Scaled
    





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
    myPerceivedRisk = beta*climatePerturbationF(tLag, tData_ts)
    return sp.norm(loc=0,scale=1).cdf(myPerceivedRisk)
    
if False: # Using function
    testData=np.array([100,110,120,100])
    perceivedRisk(0,3,testData,beta=1.0)
    

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


#### eIncrement ################################################################

def eIncrement(att, pbc, psn):
    """
    eIncrement[att_,pbc_,psn_]: rescales att and psn to -Inf to Inf and 
    then multiplies by pbc (0 to 1) to result in a increment in per 
    capita emissions
    att = attitude, pbc = perceivedBehavioralControl, psn = perceivedSocialNorm
    """
    attInv = sp.norm(loc=0.0,scale=1.0).ppf(att) # InverseCDF
    attInv[attInv==-np.inf]= min(10*attInv[attInv!=-np.inf].min(),-10) # avoid -inf 
    attInv[attInv==np.inf]= max(10*attInv[attInv!=np.inf].max(),10) # avoid +inf
    
    psnInv = sp.norm(loc=0.0,scale=1.0).ppf(psn) # InverseCDF
    psnInv[psnInv==-np.inf]= min(10*psnInv[psnInv!=-np.inf].min(),-10) # avoid -inf
    psnInv[psnInv==np.inf]= max(10*psnInv[psnInv!=np.inf].max(),10) # avoid +inf
    
    eDelIncrement = -(attInv + psnInv)*pbc
  
    return eDelIncrement
    
    
if False: # Using function
    emissionsPC=randomNormalF(5.049, 0.5, 10)
    testClimateData=np.array([100,110,120,100])
    perRisk=perceivedRisk(0,3,testClimateData,beta=1.0)
    eff=efficacyF(1)
    att=attitude(perRisk, eff)
    pbc=perceivedBehavioralControlF(1)
    psn=perceivedSocialNorm(emissionsPC)
    emissionsPC_Del = eIncrement(att, pbc, psn)
    
    perRisk=perceivedRisk(0,3,tData,beta=1.0)
    eff=efficacyF(10)
    att=attitude(perRisk, eff)
    pbc=perceivedBehavioralControlF(10)
    psn=perceivedSocialNorm(pcE)
    eIncrement(att, pbc, psn)
    
    
    psnInv[psnInv==-inf]= min(10*psnInv[psnInv!=-inf].min(),-10)


#### updatePCEmissions ################################################################

def updatePCEmissions(pcE, eff, pbc, tData,percepWindowSize,riskSens=1.0):
    """
    updatePCEmissions calculates a del pcE and then adds this to current pcE to
    return new pcE in 
    """
    # climateDat=testClimateData; yearCurrent=0; percepWindowSize=3; riskSens=1.0
    psn=perceivedSocialNorm(pcE) #
    risk = perceivedRisk(percepWindowSize, tData, riskSens)
    att = attitude(risk, eff)
    # emissionsPC_Index = pcEmissionsToIndex(emissionsPC, mean=emissionsPC.mean(), sd=emissionsPC.std())
    pcE_Del = eIncrement(att, pbc, psn)
    pcE_New = pcE_Del + pcE
  
    return pcE_New
    
    
if False: # Using function
    emissionsPC=randomNormalF(5.049, 0.5, 10)
    testClimateData=np.array([100,110,120,100])
    perRisk=perceivedRisk(0,3,testClimateData,beta=1.0)
    eff=efficacyF(10)
    att=attitude(perRisk, eff)
    pbc=perceivedBehavioralControlF(10)
    psn=perceivedSocialNorm(emissionsPC)
    emissionsPC_Del = eIncrement(att, pbc, psn)
    emissionsPC_New = emissionsPC_Del + emissionsPC
    updatePCEmissions(emissionsPC, eff, pbc, testClimateData, 0,3,riskSens=1.0)

#### iterateOneStep ############################################################


def iterateOneStep(pcE_ts, tData_ts, co2_ts, eff, pbc, popN,percepWindowSize=3,riskSens=1.0):
    """
    Updates atm CO2, temperature and per capita emissions for one step (one year).
    """
    pcE_updated=updatePCEmissions(pcE_ts[:,-1], eff, pbc,tData_ts,percepWindowSize,riskSens)
    pcE_updated=np.atleast_2d(pcE_updated).transpose()
    pcE_vector=np.concatenate((pcE_ts, pcE_updated),axis=1)
    co2Del_ppm=perCapitaEmissionsToDelPPM(pcE_updated, popN)
    co2_updated = np.array([co2Del_ppm + co2_ts[-1]])  # adds to last element of co2Current
    co2_vector = np.concatenate( [co2_ts, co2_updated] )
    rf = computeRF(co2_vector)
    tDel=compute_deltaT(rf)
    t_updated = np.array([tDel[-1] + tData_ts[-1]])  # adds to last element of co2Current
    t_vector = np.concatenate( [tData_ts, t_updated] )
        
    return pcE_vector,t_vector,co2_vector
    
    
if False: # Using function
    
    co2_ts=np.linspace(290, 298, 4) # co2 initial values
    popTotal=7130010000 # Wolfram: QuantityMagnitude[CountryData["World", "Population"]]
    popN=popIntoNgroups(popTotal,nGroups=10)
    pcE_ts=randomNormalF(5.049, 0.5, 10)
    pcE_ts=np.atleast_2d(pcE_ts).transpose()
    tData_ts=np.array([0,0.1,0.2,0.1]) # temperature initial values
    
    eff=efficacyF(10)
    pbc=perceivedBehavioralControlF(10)
    yearCurrent=0
    percepWindowSize=3
    riskSens=1.0
    
    pcE_ts,tData_ts,co2_ts = iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
        percepWindowSize=3,riskSens=1.0)
        
    pcE_ts,tData_ts,co2_ts = iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
        percepWindowSize=3,riskSens=1.0)
        
    pcE_ts,tData_ts,co2_ts = iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
        percepWindowSize=3,riskSens=1.0)
        
    
    
###### iterating function ################################################################

if False:
    
    for i in range(10):
        print i
        pcE_ts,tData_ts,co2_ts = iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
            percepWindowSize=3,riskSens=1.0)
        
    # tData_ts.plot()
    import matplotlib.pyplot as plt
    hold(False)
    
    plt.plot(tData_ts)
    plt.xlabel('Year')
    plt.ylabel('T')
    
    plt.plot(co2_ts)
    plt.xlabel('Year')
    plt.ylabel('ppm')
    
    plt.plot(pcE_ts)
    plt.xlabel('Year')
    plt.ylabel('ppm')
    
    


