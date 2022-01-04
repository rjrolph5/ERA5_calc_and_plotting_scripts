
# This script creates a mask over grid cells & timesteps where the sea ice 
# concentration exceeds a user-specified threshold

import numpy as np
from netCDF4 import Dataset
import xarray as xr

# assign threshold defining open water or sea ice cover
sicn_threshold = 0.15

# Define paths
data_basepath = '/permarisk/data/ERA_Data/ERA5_becca/'
datapath_sicn = data_basepath + 'era5_data/sea_ice_area_fraction/'
datapath_sicn_mask = data_basepath + 'era5_sea_ice_mask/'

# Read sea ice raw data files (ERA5) and create boolean sea ice mask
for year in np.arange(1979,2021):
	print(year)

	ifile_sicn = datapath_sicn + 'ERA5_' + str(year) + '_sea_ice_area_fraction.nc'

	ds_sicn_current_loop_year = xr.open_dataset(ifile_sicn)

	da_sicn = ds_sicn_current_loop_year.siconc


	# create mask during times of sea ice cover, based on threshold above
	mask_sicn = xr.where(da_sicn > sicn_threshold, 1, 0) # true where sea ice cover

	# save the mask
	ofile_mask = datapath_sicn_mask + 'ERA5_' + str(year) + '_sea_ice_mask_true_above_' + str(int(sicn_threshold*100)) + 'percent.nc'
	mask_sicn.to_netcdf(ofile_mask)
