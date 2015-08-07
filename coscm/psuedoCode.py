
perceived behavioral control

pbc = n random values on scale (0,1) # 1 leads to increased per capitat emissions, 0 leads to decreased per capitat emissions




efficiacy

pbc = n random values on scale (0,1) # 1 leads to increased per capitat emissions, 0 leads to decreased per capitat emissions





perceived social norm # scaled 0 to 1

group differences= xVect-xVect.mean()

scaled group differences= on uniform 1 e.g., sp.uniform(loc=-1.0,scale=2.0).cdf(xDiff) # switch to normal cdf?





Climate anomaly

climate anomaly = temp of current year - mean(temp in 3 previous years)
    #perturbation=tData_ts[-1]-tData_ts[-2:(-2-windowWidth):-1].mean()




Perceived Risk


perceived Risk = riskSensitivity * climate anomaly
    #myPerceivedRisk = beta*climatePerturbationF(tLag, tData_ts)
    #    return sp.norm(loc=0,scale=1).cdf(myPerceivedRisk)






Attitude # 1 is really concerned, 0 is not concerned

    perceived risk rescaled = sp.norm(loc=0,scale=1).ppf(perceivedRisk) # inverseCDF rescales to -inf to inf
    myAttitudeInv = perceived Risk rescaled * efficacy # larger value -> more motivation to reduce pcE
    attitude = 1 - sp.norm(loc=0,scale=1).cdf(myAttitudeInv) # reversing so 0 -> lower pcE, 1-> more pcE
    

increment per capita emissions

    attInv = sp.norm(loc=0.0,scale=1.0).ppf(att) # InverseCDF to rescale to -inf to inf
    psnInv = sp.norm(loc=0.0,scale=1.0).ppf(psn) # InverseCDF to rescale to -inf to inf
    eDelIncrement = -(attInv + psnInv) * pbc
    pcE_New = pcE_Del + pcE
    if pcE < 0.1 initial pcE ->0.1 initial pcE # add this code to be consistent with Travis' model
    













