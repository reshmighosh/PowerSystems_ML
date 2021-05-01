import numpy as np
import pandas as pd
import netCDF4 as nk
import os
import glob
import sys

path = '/scratch/mtcraig_root/mtcraig1/shared_data/reshmig_chp2/data/processed/'
os.chdir(path)
files = sorted(glob.glob("*.nc"))
#print(files)

# Create numpy arrays for Dallas
lat = 32.7767
lon = -96.7970
test = str(sys.argv[-1])

train_array = []
test_array = []

def concat_Dallas_temp(files, dallas_lat, dallas_lon, test):
   if test == 'True':
      for filename in files:
         if filename not in ('processedMERRAercot2015.nc', 'processedMERRAercot2016.nc', 'processedMERRAercot2017.nc', 'processedMERRAercot2018.nc', 'processedMERRAercot2019.nc'):
            print("printing test filename", filename)
            converted_nk = nk.Dataset(filename)
            temp = converted_nk.variables['T2M']
            temp_celcius = np.array(temp) - 273
            temp_lats = np.array(converted_nk.variables['lat'][:])
            temp_lons = np.array(converted_nk.variables['lon'][:])
            lat_interest = (np.abs(dallas_lat - temp_lats)).argmin()
            lon_interest = (np.abs(dallas_lon - temp_lons)).argmin()
            temp_final = temp_celcius[lat_interest *lon_interest, :, :].ravel()
            test_array.append(temp_final)
      finaltest_array = np.concatenate(test_array, axis = 0)
      print(len(finaltest_array), "length of all test arrays")
      return finaltest_array
   elif test == 'False':
      for filename in files:
         #print("train mode")
         if filename in ('processedMERRAercot2015.nc', 'processedMERRAercot2016.nc', 'processedMERRAercot2017.nc ','processedMERRAercot2018.nc', 'processedMERRAercot2019.nc'):
            print("printing train filename", filename)
            converted_nk = nk.Dataset(filename)
            temp = converted_nk.variables['T2M']
            temp_celcius = np.array(temp) - 273
            temp_lats = np.array(converted_nk.variables['lat'][:])
            temp_lons = np.array(converted_nk.variables['lon'][:])
            lat_interest = (np.abs(dallas_lat - temp_lats)).argmin()
            lon_interest = (np.abs(dallas_lon - temp_lons)).argmin()
            temp_final = temp_celcius[lat_interest *lon_interest, :, :].ravel()
            train_array.append(temp_final)
      finaltrain_array = np.concatenate(train_array, axis = 0)
      print(len(finaltrain_array), "length of all train arrays")
      return finaltrain_array
concat_Dallas_temp(files, lat, lon, test)







    







