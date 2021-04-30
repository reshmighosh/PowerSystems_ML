# import libraries

import os
import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.graphics.tsaplots as tsa
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install netCDF4 if netCDF4 is not installed
import netCDF4 as nk
import glob

year_choice = sys.argv[1]


demand_folder = ""
final = list()
output_dir = "" # save results here
tbins = [10, 15, 20, 30] # temperature bins for piecewise linear regression

# read all demand files
files = []
for dir, folder, file_path in os.walk(demand_folder):
  for file in sorted(file_path):
    if file.endswith('.xlsx') or file.endswith('.xls'):
      files.append(os.path.join(dir, file))


def get_ERCOT_demand(files):
    for file in files:
        df = pd.read_excel(file)
        if file == path + ercot_demand_path + 'Native_Load_2017.xlsx':
            df.iloc[7393, 0] = df.iloc[7393, 0][:-4]
        df.columns = ['Hour_End', 'COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTHERN',
       'SOUTH_C', 'WEST', 'ERCOT']
       final_dfs.append(df)
    final_df = pd.concat(final_dfs, axis = 0)
    return final_df

def get_temperature():
Dallas_lat = 32.7767
Dallas_lon = -96.7970
global years

def importERCOTtemp(year, start_year, end_year, lat, lon):
    years = list(np.arange(start_year, end_year+1, 1))
    num_periods = end_year - start_year + 1
    
    final_temp = dict()
    for year_int in years:
        temp_file = nk.Dataset(path + ercot_temp_path + 'processedMERRAercot' + str(year_int) + '.nc')
        temp = temp_file.variables['T2M']
        powGen_lats = np.array(temp_file.variables['lat'][:])
        powGen_lons = np.array(temp_file.variables['lon'][:])
        lat_interest = np.where(np.abs(Dallas_lat - powGen_lats).argmin())
        #unique, counts = np.unique(np.array(temp), return_counts= True)
        #frequencies = np.asarray((unique, counts)).T
        #print(f'unique values for year {year} is {frequencies}')
        temp_celcius = np.array(temp) - 273 
        lat_interest = (np.abs(Dallas_lat - powGen_lats)).argmin()
        lon_interest = (np.abs(Dallas_lon - powGen_lons)).argmin()
        temp_final = temp_celcius[lat_interest *lon_interest, :, :].ravel()
        final.append(temp_final)
        final_temp[str(year)] = temp_final
    
    #large array
    final_array = np.concatenate(temp_final, axis = 0)
    final_df = pd.DataFrame(final_array, columns = ['Temperature'])

    # converting final temperature to dataframe
    final_temp = pd.DataFrame(data = final_temp[str(year)], columns = ['Temperature'])
    time = pd.date_range('01-01-2017', periods = 8760*num_years, freq = 'H')

    final_df['Hour'] = time
    final_df.set_index('Hour')
    final_temp['Hour'] = time
    final_temp.set_index('Hour')

    return final_df, final_temp

  #final_temp_reshaped = final_temp.ravel()

df, year_temp = importERCOTtemp(2017, 2003, 2020, Dallas_lat, Dallas_lon)

tempbins = [10, 15, 20, 30]
def createTempBins(temps, tempsbin, year, start_year, end_year, lat, lon):
  #temps = importERCOTtemp(year, start_year, end_year, lat, lon)
  nComps = len(tempsbin) + 1
  tempsBinned = pd.DataFrame(0, index = temps.index, columns = ['c' + str(i) for i in range(len(tempsbin))])
  for i in range(1,len(tempsbin)):
    rows = (temps['Temperature']<tempsbin[i]) & (temps['Temperature']>=tempsbin[i-1])
    #print(tempsbin[i], tempsbin[i-1], "printing first case")
    tempsBinned.loc[rows,'c' + str(i)] = temps['Temperature'][rows] - tempsbin[i-1]
    rows = (temps['Temperature']>=tempsbin[i])
    #print(tempsbin[i], "printing second case")
    tempsBinned.loc[rows,'c'+str(i)] = tempsbin[i] - tempsbin[i-1]
    rows = (temps['Temperature']>tempsbin[-1])
    tempsBinned.loc[rows,'c'+str(nComps-1)] = temps['Temperature'][rows] - tempsbin[-1]
    #print(tempsbin[-1], (nComps -1 ), "printing third case")
    #
    print(tempsBinned)

def create_FE():
    


if __name__ = "main":
df, year_temp = importERCOTtemp(year_choice, 2003, 2020, Dallas_lat, Dallas_lon)





