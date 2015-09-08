
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SoCCo as sc


# why this error?

#/Users/brianbeckage/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/pytz/__init__.py:29: 
#    UserWarning: Module canopy_debugger was already imported from /Users/brianbeckage/Library/Enthought/Canopy_64bit/
#    System/lib/python2.7/site-packages/canopy_debugger/__init__.pyc, but /Applications/Canopy.app/appdata/
#    canopy-1.5.5.3123.macosx-x86_64/Canopy.app/Contents/lib/python2.7/site-packages is being added to sys.path
#  from pkg_resources import resource_stream




### projecting temperature for one time step ################################### 


if False: ### WORKS!
    co2 = np.linspace(290, 300, 11)
    # rf = computeRF(co2)
    sc.computeRF(co2) 
    

### initialize and update coupled climate and social model  ####################


if False: ### WORKS!
    emissionsPC=sc.randomNormalF(5.049, 0.5, 10)
    testClimateData=np.array([100,110,120,100])
    perRisk=sc.perceivedRisk(3,testClimateData,1.0)
    eff=sc.efficacyF(10)
    att=sc.attitude(perRisk, eff)
    pbc=sc.perceivedBehavioralControlF(10)
    psn=sc.perceivedSocialNorm(emissionsPC)
    emissionsPC_Del = sc.eIncrement(att, pbc, psn)
    emissionsPC_New = emissionsPC_Del + emissionsPC
    sc.updatePCEmissions(emissionsPC_New, eff, pbc, testClimateData, 3,riskSens=1.0)



if False: ### WORKS!
    co2_ts=np.linspace(290, 300, 4)
    popTotal=7130010000 # Wolfram: QuantityMagnitude[CountryData["World", "Population"]]
    popN=sc.popIntoNgroups(popTotal,nGroups=10)
    pcE_ts=sc.randomNormalF(5.049, 0.5, 10)
    pcE_ts=np.atleast_2d(pcE_ts).transpose()
    tData_ts=np.array([0,0.1,0.2,0.1]) # temperature
    
    eff=sc.efficacyF(10)
    pbc=sc.perceivedBehavioralControlF(10)
    # yearCurrent=0
    percepWindowSize=3
    riskSens=1.0
    
    pcE_ts,tData_ts,co2_ts = sc.iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
        percepWindowSize=3,riskSens=1.0)
        
    pcE_ts,tData_ts,co2_ts = sc.iterateOneStep(pcE_ts,tData_ts, co2_ts, eff, pbc,popN,
        percepWindowSize=3,riskSens=1.0)
        


### Updating model n steps #####################################################

    co2_ts=np.linspace(290, 300, 4)
    popTotal=7130010000 # Wolfram: QuantityMagnitude[CountryData["World", "Population"]]
    popN=sc.popIntoNgroups(popTotal,nGroups=10)
    pcE_ts=sc.randomNormalF(5.049, 0.5, 10)
    pcE_ts=np.atleast_2d(pcE_ts).transpose()
    tData_ts=np.array([0,0.1,0.2,0.1]) # temperature
    eff=sc.efficacyF(10)
    pbc=sc.perceivedBehavioralControlF(10)
    percepWindowSize=3
    riskSens=1.0
    
    pcE_ts,tData_ts, co2_ts=sc.iterateNsteps(pcE_ts,tData_ts, co2_ts, 5, eff, pbc,popN,
            percepWindowSize,riskSens)

        
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
    
    





### Code Testing ###############################################################
if False:

### using functions



### Importing real data ################
    
    # CO2 data
    co2Data=pd.read_excel("co2-atmospheric-mlo-monthly-scripps.xls", 
                        sheetname='Monthly & Annual CO2 Data', skiprows=6)
    co2Data = co2Data.set_index('Year')
    co2Data.head()
    co2Data.shape
    co2A = co2Data[[u'Annual Average']] # keeping only annual means
    #co2A = co2A.astype(np.float64)
    co2A.shape
    co2A.columns=['annual CO2']
    co2A.index=pd.to_datetime(pd.Series(co2A.index), format="%Y") # unit=D) #infer_datetime_format=True) # unit=D
    co2A.head()
    co2A.plot()                 

# global temperature data

    giss_temp = pd.read_table("http://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.txt", sep="\s+", skiprows=7,
                            skip_footer=11, engine="python")
    giss_temp = giss_temp.set_index('Year')
    giss_temp.head()
    tempA = giss_temp[[u'J-D']] # keeping only annual means
    tempA = tempA.drop("Year")
    tempA = tempA.where(tempA != "****", np.nan)
    tempA = tempA.astype(np.float64)
    tempA = tempA.astype(np.float64)/100 # rescaling T
    tempA.index=pd.to_datetime(pd.Series(tempA.index),format="%Y")
    tempA.columns=['annual T']
    tempA.plot()


### Combining series into a dataframe
    climateDat=pd.concat([tempA, co2A], axis=1)
    climateDat.head()
    climateDat=climateDat.dropna() # removing NAs
    climateDat.head()
    climateDat.info()
    climateDat.plot()
    climateDat.plot(subplots=True, figsize=(16, 12))
    # climateDat.to_csv('climateDat.csv')


### Projecting temperature change using obs CO2

    rf=computeRF(climateDat[['annual CO2']])
    annualPred_T=compute_deltaT(rf)
    climateDat.head()
    climateDat.columns
    climateDat['Annual delT Pred']=compute_deltaT(rf)
    climateDat['Annual T Pred']=climateDat['Annual delT Pred'] +climateDat['annual T'].iloc[0:10].mean()
        #+climateDat['annual T'].iloc[0]
    climateDat.head()
    climateDat.plot(subplots=True, figsize=(16, 12))



### Plotting comparison to data ################################################

    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(climateDat.index,climateDat[['annual CO2']], 'b-')
    ax1.set_ylabel('ppm', color='b')
    ax2 = ax1.twinx()
    ax2.plot_date(climateDat.index,climateDat[['Annual T Pred']], 'r.')       
    ax2.plot_date(climateDat.index,climateDat[['annual T']], 'r-^')
    ax2.set_ylabel('T', color='r')
    ax1.set_xlabel('year')
    ax1.legend(['annual CO2'])
    ax2.legend(['Annual T pred','annual T obs'])







### scratch code
    
    time = range(2000,2011)
    co2 = np.linspace(290, 300, 11)
    rf = computeRFv(co2)
    len(time)
    len(co2)
    len(rf)
    compute_deltaT(time, rf)
    
    print np.any(tempA==np.nan) # no missing values
    # tempA[tempA==np.nan]
    # giss_temp = giss_temp.where(giss_temp != "****", np.nan)
    # np.isclose(giss_temp,-999.000)
    
    a = array([1,2,3])
    b = array([4,5,6])
    np.vstack((a,b)).transpose()
    np.concatenate((a,b)).transpose() # not quite what I'm looking for
    # np.hstack((a,b))
    
    


    





