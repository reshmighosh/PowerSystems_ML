
#Regress temperature vs. demand for all years

from ImportERCOTDemand import importHourlyERCOTDemand
import statsmodels.api
import statsmodels.graphics.tsaplots as tsa
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
# from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
import numpy as np
import pandas as pd
import os, h5py

################################################################################
def regressTempVsLoad(doPlots,tBins,incYrFE,centerOrCity,years,toScaleDemand,hpc):
    resultsDir = 'TempAndLoad'
    #Import hourly load data for all years.
    zonalDemand = None 
    for yr in years:
        if yr != 2001: #missing data
            zonalDemandYr = importHourlyERCOTDemand(yr,'remote',toScaleDemand) #remote is runLoc
            # print(zonalDemandYr)
            if zonalDemand is None: zonalDemand = zonalDemandYr.copy()
            else: zonalDemand = pd.concat([zonalDemand,zonalDemandYr])
    zonalDemand.drop(['ercot'],axis=1,inplace=True)
    #Import wz lat/lon coords to use
    wzCoords = getWZLatLongs(centerOrCity)
    #Set formula
    regFormula = setRegFormula(tBins,incYrFE)
    #Run regression across years for each wz
    fittedRegsByWz,regStats,loadActualAndPred = dict(),[['zone','rmse','rsquared']],None
    allCoeffs,ctr = None,1
    # maxt,mint = -100,100
    for wz in zonalDemand:
        print('Weather zone:',wz)
        load = zonalDemand[wz]  
        #Import WRF temps
        temps = importHistoricWRFTemps(wzCoords[wz][0],wzCoords[wz][1],years,'TX',hpc)
        #Save max & min
        # if temps.max()>maxt: maxt=temps.max()
        # if temps.min()<mint: mint=temps.min()
        #Create DF and run regression
        tempsBinned,regDf,regFit = runRegression(temps,tBins,load,regFormula)
        # print(regFit.summary())
        #Save regression results
        fittedRegsByWz[wz] = regFit
        regResults = pd.DataFrame({'Params':regFit.params,'Errors':regFit.bse},
            index=regFit.params.index)
        if allCoeffs is None: 
            allCoeffs = pd.DataFrame({wz:regFit.params},index=regFit.params.index)
        else: 
            allCoeffs[wz] = regFit.params
        #Calc RMSE
        dep = regFormula.split('~')[0].strip()
        rmse,predictedLoad = calcRMSEOfPredictedVals(regDf,regFit,dep)
        regStats.append([wz,rmse,regFit.rsquared])
        #Save actual and predicted load
        if loadActualAndPred is None: 
            loadActualAndPred = pd.DataFrame({wz:regDf['load'].values,
                            wz+'pred':predictedLoad},index = regDf['load'].index)
        else: 
            loadActualAndPred[wz] = regDf['load'].values
            loadActualAndPred[wz+'pred'] = predictedLoad
        #Save data
        tempsBinned.to_csv(os.path.join(resultsDir,'tempBinned' + wz + '.csv'))
        regDf.to_csv(os.path.join(resultsDir,'regData' + wz + '.csv'))
        regResults.to_csv(os.path.join(resultsDir,'RegCoeffs' + wz + '.csv'))
        #Make plots
        if doPlots == True: makePlots(regDf,regResults,predictedLoad,wz,tBins,ctr,resultsDir)            
        ctr += 1
    print('Regression results:',regStats)
    write2dListToCSV(regStats,os.path.join(resultsDir,'regStats.csv'))
    loadActualAndPred.to_csv(os.path.join(resultsDir,'actualAndPredLoad.csv'))
    allCoeffs.to_csv(os.path.join(resultsDir,'regressionCoeffs.csv'))
    if doPlots == True: plt.show()
    return allCoeffs,regFormula,fittedRegsByWz,wzCoords

#Create DF then run regression. Inputs: reg formula, load (pd series),
#temps (pd series), tbins (np array)
def runRegression(temps,tBins,load,regFormula):
    #Create reg DF and run regression
    regDf,tempsBinned = createRegDf(tBins,temps,load)
    regFit = sm.ols(formula=regFormula,data=regDf).fit()
    return tempsBinned,regDf,regFit
################################################################################

################################################################################
def getWZLatLongs(centerOrCity):
    if centerOrCity == 'center': locs = getZoneCenters()
    else: locs = getCityLocs()
    locs = replaceWZNames(locs)
    return locs

def getZoneCenters():
    dataDir = 'Data\\ERCOTShapeFiles'
    fs = os.listdir(dataDir)
    zonePolys = dict()
    for f in fs:
        if '.shp' in f:
            zone,fname = f.split('_')[0].lower(),f.split('.')[0]
            sf = shapefile.Reader(os.path.join(dataDir,fname))
            shapes = sf.shapes()
            points = shapes[0].points
            poly = Polygon(points)
            zonePolys[zone] = poly.centroid.coords
    return zonePolys

def getCityLocs():
    cities = pd.read_csv(os.path.join('Data','WeatherZoneToCity.csv'))
    cityLocs = dict()
    for row in cities.index:
        wz = cities.loc[row,'WeatherZone'].replace(' ','')
        cityLocs[wz] = (cities.loc[row,'Lat'],cities.loc[row,'Long'])
    return cityLocs

def replaceWZNames(locs):
    reps = {'northcentral':'north_c','southcentral':'south_c','south':'southern',
            'farwest':'far_west'}
    newlocs = dict()
    for wz in locs:
        if wz in reps: newlocs[reps[wz]] = locs[wz]
        else: newlocs[wz] = locs[wz]
    return newlocs
################################################################################

################################################################################
def importHistoricWRFTemps(lat,lon,years,modelName,hpc):
    allTemps = list()
    for yr in years:
        tprh = 'TPRH_' if modelName != 'TX' else ''
        if hpc: filePath = '/projects/ldrdclimate/Base_Period/humidity_temp_pres'
        else: filePath = os.path.join('Data','MetVars')
        metVars = h5py.File(os.path.join(filePath,modelName + '_' + tprh + str(yr) + '.h5'),'r')
        lats,lons = metVars['meta']['latitude'],metVars['meta']['longitude']    
        dists = calcHaversineDist(lat,lon,lats,lons)
        closestIdx = np.argmin(dists)
        temps = metVars['temperature_2m'][:,closestIdx]/100
        temps = pd.Series(temps,index=[pd.to_datetime(dt.decode()) for dt in metVars['time_index']])
        allTemps.append(temps)
    return pd.concat(allTemps)

def calcHaversineDist(genlat,genlon,lats,lons):
    genlat,genlon = np.radians(genlat),np.radians(genlon)
    lats,lons = np.radians(lats),np.radians(lons)
    R = 6371 #km; mean Earth radius
    diffLat = lats - genlat
    diffLong = lons - genlon
    h = (np.sin(diffLat/2))**2 + np.cos(lats) * np.cos(genlat) * (np.sin(diffLong/2))**2
    c = 2 * np.arcsin(np.sqrt(h))
    return R*c #dist in km
################################################################################

################################################################################
#Create nxm array where m = # temperature bins. Inputs: pd series of temperature,
#list of temperature break points of piecewise linear funtion.
def createTempComponents(temps,tempBins):
  # replace NA with -9999
  temps.fillna(-9999,inplace=True)
  #Set # rows, bounds, components
  nRows,nBounds = temps.shape[0],len(tempBins)
  nComps = nBounds+1
  #Create binned array
  tempsBinned = pd.DataFrame(0,index=temps.index,columns=['c' + str(i) for i in range(nComps)])
  #Assign first component
  tempsBinned.loc[temps<tempBins[0],'c0'] = temps[temps<tempBins[0]]
  tempsBinned.loc[temps>=tempBins[0],'c0'] = tempBins[0]
  for i in range(1,len(tempBins)):
    rows = (temps<tempBins[i]) & (temps>=tempBins[i-1])
    tempsBinned.loc[rows,'c' + str(i)] = temps[rows] - tempBins[i-1]
    rows = (temps>=tempBins[i])
    tempsBinned.loc[rows,'c'+str(i)] = tempBins[i] - tempBins[i-1]
  rows = (temps>tempBins[-1])
  tempsBinned.loc[rows,'c'+str(nComps-1)] = temps[rows] - tempBins[-1]
  return tempsBinned
################################################################################

################################################################################
def createRegDf(tBins,temps,load):
    #Divide temperature into bins
    tempsBinned = createTempComponents(temps,tBins)
    #Create reg df w/ all data
    regDf = tempsBinned.copy()
    regDf['season'] = (regDf.index.month%12+3)//3 - 1
    regDf['year'] = regDf.index.year
    regDf['hour'] = regDf.index.hour
    regDf['dayType'] = regDf.index.weekday//5 #weekend vs weekday; 5 is value of Saturday
    #Add in load data if fitting df
    regDf['load'] = load
    #Combine day, hour, season into single series of fixed effects
    regDf['hourDaySeason'] = (regDf['hour'].astype(str) + regDf['dayType'].astype(str) + 
                                regDf['season'].astype(str))
    regDf.dropna(inplace=True) #elim rows w/ missing load data
    return regDf,tempsBinned

def setRegFormula(tBins,incYrFE):
    tempStrs = ''
    for i in range(len(tBins)+1): tempStrs += 'c' + str(i) + ' + '
    return 'load ~ ' + tempStrs + ('C(year) +' if incYrFE is True else '') + 'C(hourDaySeason)'

def calcRMSEOfPredictedVals(df,fitReg,yName):
    predictedLoad = fitReg.predict(df)
    actualPrices = df[yName]
    return calcRmse(predictedLoad,actualPrices),predictedLoad

#Calc RMSE b/wn 2 pandas series
def calcRmse(s1,s2):
    return (((s1-s2)**2).mean())**.5
################################################################################

################################################################################
def makePlots(regDf,regResults,predictedLoad,wz,tBins,ctr,resultsDir):
    #Plotting parameters
    dts = [pd.date_range('1/1/2003','1/8/2003',freq='H'),
            pd.date_range('4/1/2003','4/8/2003',freq='H'),
            pd.date_range('7/1/2003','7/8/2003',freq='H'),
            pd.date_range('10/1/2003','10/8/2003',freq='H')]

    #Plot timeseries of LDC of actual versus predicted load
    temp = pd.DataFrame({'actual':regDf['load'].values,'pred':predictedLoad},
            index = regDf.index) 
    plt.figure(1,figsize=(8,8))
    actualSorted = temp['actual'].sort_values(ascending=False)
    predSorted = temp['pred'].sort_values(ascending=False)
    ax = plt.subplot(4,2,ctr)
    ax.plot(actualSorted.values,label='Actual')
    ax.plot(predSorted.values,label='Predicted')
    plt.title(wz,y=.8)
    if ctr % 2 == 1: plt.ylabel('Load (MW)')
    if ctr >= 7: plt.xlabel('Hour')
    if ctr == 8: plt.legend()
    plt.savefig(os.path.join(resultsDir,'LDCs.png'),dpi=300,
        transparent=True, bbox_inches='tight', pad_inches=0.1)        
    
    #Plot time series of actual versus predicted load
    # plt.figure(figsize=(8,10))
    # for i in range(len(dts)):
    #     dt,ax = dts[i],plt.subplot(2,2,i+1)
    #     actual,pred = temp['actual'].loc[dt],temp['pred'].loc[dt]
    #     actual.plot(ax=ax,label='Actual')
    #     pred.plot(ax=ax,label='Predicted')
    #     if i == 0 or i == 2: plt.ylabel('Load (MW)')
    #     if i == 0: plt.title(wz)
    # plt.legend()
    # plt.savefig(os.path.join(resultsDir,'TimeSeriesActualVPredicted' + wz + '.png'),dpi=300,
    #     transparent=True, bbox_inches='tight', pad_inches=0.1)        
    
    #Plot residualized load versus fitted load. Residualized load
    #is load minus all non-temperature effects; fitted load is T effects and intercept. 
    #Create residualized load
    residLoad = regDf.copy()
    residLoad['yearLabel'] = 'C(year)[T.' + residLoad['year'].astype(str) + ']'
    residLoad = residLoad.merge(regResults,how='left',left_on='yearLabel',right_index=True)
    residLoad['hdsLabel'] =  'C(hourDaySeason)[T.' + residLoad['hourDaySeason'].astype(str) + ']'
    residLoad = residLoad.merge(regResults,how='left',left_on='hdsLabel',right_index=True)
    residLoad['Params_x'].fillna(0,inplace=True)
    residLoad['Params_y'].fillna(0,inplace=True)
    residLoad['resid'] = residLoad['load']-residLoad['Params_x']-residLoad['Params_y']
    residLoad['temp'] = 0
    for i in range(len(tBins)+1): residLoad['temp'] += residLoad['c' + str(i)]
    #Create fitted load lines
    xmin,xmax = -5,40
    temps = pd.Series([t for t in list(range(xmin,xmax))])
    tempsBinned = createTempComponents(temps,tBins)
    fittedTs = pd.Series(0,index=tempsBinned.index)
    for i in range(len(tBins)+1): 
        fittedTs += tempsBinned['c'+str(i)]*regResults.loc['c'+str(i),'Params']
    fittedTs += regResults.loc['Intercept','Params']
    fittedTs.index = temps
    #Plot
    plt.figure(figsize=(2.8,2.2))
    ax = plt.subplot(111)
    plt.hist2d(residLoad['temp'].values,residLoad['resid'].values,bins=100,cmin=1)
    plt.colorbar()    
    fittedTs.plot(ax=ax,color='red',lw=2)
    ax.set_xlim([xmin,xmax])
    plt.title(wz,fontsize=10)
    plt.ylabel('Residual Load (MW)',fontsize=10)
    plt.xlabel('Temperature (degrees C)',fontsize=10)
    # plt.savefig(os.path.join(resultsDir,'ResidVFittedLoad' + wz + '.png'),dpi=600,
    #     transparent=True, bbox_inches='tight', pad_inches=0.1)        
################################################################################

if __name__=='__main__':
    doPlots = True
    incYrFE,toScaleDemand = True,False
    centerOrCity = 'city' #whether to select WRF var as centroid of poly or nearest city
    years = list(range(1996,2001)) + list(range(2002,2006))
    tBins = np.array([10,15,20,25])
    allCoeffs,regFormula,fittedRegsByWz,wzCoords = regressTempVsLoad(doPlots,tBins,
                                        incYrFE,centerOrCity,years,toScaleDemand,False)