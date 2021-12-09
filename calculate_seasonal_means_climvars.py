
# Calculate the seasonal means from hourly ERA5 dataset of user-specified input variables.

from netCDF4 import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

datapath = '/permarisk/data/ERA_Data/ERA5_becca/'

# load a climate variable file
climvar = 'era5_wind_speed' # '2m_temperature', 'sea_ice_area_fraction'
var_shortname = 'wind_speed_10m' # 't2m', 'siconc'
units = 'm/s' # 'K', 'frac'

seasonal_mean_variable_datapath = datapath  + 'era5_seasonal_means/' + var_shortname + '/'
# check if the path exists, if not, create it.
Path(seasonal_mean_variable_datapath).mkdir(parents=True, exist_ok=True)

year_range = np.arange(1979,2021)

for year in year_range:
	print(str(year))
	if climvar == 'era5_wind_speed': # era5 wind speed is not in the same folder as the raw data downloaded from copernicus, since it had to be calculated from the raw vector data.
		climvar_ifile_current_loop_year = datapath + 'era5_wind_speed/ERA5_' + str(year) + '_wind_speed_10m.nc'
		climvar_dataset_current_loop_year = xr.open_dataset(climvar_ifile_current_loop_year)
	else:
		climvar_ifile_current_loop_year = datapath + 'era5_data/' + climvar + '/ERA5_' + str(year) + '_' + climvar + '.nc'
		climvar_dataset_current_loop_year = xr.open_dataset(climvar_ifile_current_loop_year)

	if year != 1979: # we dont have ERA5 data before 1979.
		if climvar == 'era5_wind_speed':
			climvar_ifile_preceding_year = datapath + 'era5_wind_speed/ERA5_' + str(year - 1) + '_wind_speed_10m.nc'
			climvar_ds_preceding_year = xr.open_dataset(climvar_ifile_preceding_year)
		else:
			climvar_ifile_preceding_year = datapath + 'era5_data/' + climvar + '/ERA5_' + str(year - 1) + '_' + climvar + '.nc'
			climvar_ds_preceding_year = xr.open_dataset(climvar_ifile_preceding_year)

		seasonal_mean_annual_filename = seasonal_mean_variable_datapath + 'seasonal_mean_dec' + str(year-1) + 'JF_' + 'MAM_' + 'JJA_' + 'SON_' + str(year) + '_' + var_shortname + '.nc'

	if year == 1979:
		climvar_ds_preceding_year = np.nan
		seasonal_mean_annual_filename = seasonal_mean_variable_datapath  + 'seasonal_mean_MAM_' + 'JJA_' + 'SON_' + str(year)  + '_' + var_shortname + '.nc' # different filename bc 1979 does not include DJF seasonal mean.

	# group by seasons and take the mean of each season
	ds_mean_grouped_by_season = climvar_dataset_current_loop_year.groupby('time.season').mean(skipna=True)

	# exclude DJF because december here was taken from the same year as january, and by that definition, those two months correspond to two different years in terms of seasons
	ds_mean_grouped_by_season_excluding_DJF = ds_mean_grouped_by_season.drop_sel(season=['DJF'])
	# add the year coordinate (for DJF, the labelled year is the one corresponding to JF)
	ds_mean_grouped_by_season_excluding_DJF = ds_mean_grouped_by_season_excluding_DJF.assign_coords(year=year)

	if year == 1979: # save the seasonal means excluding DJF for 1979
		ds_mean_grouped_by_season_excluding_DJF.to_netcdf(seasonal_mean_annual_filename)

	# manually calcluate the DJF average by extracting feb and jan from climvar_ifile_current_loop_year, and using december from climvar_ifile_preceding_year.
	if year != 1979:
		ds_current_loop_year_Jan_Feb = climvar_dataset_current_loop_year.sel(time=climvar_dataset_current_loop_year.time.dt.month.isin([1,2]))
		ds_prev_loop_year_Dec = climvar_ds_preceding_year.sel(time=climvar_ds_preceding_year.time.dt.month.isin([12]))
		ds_DJF_incl_dec_from_prev_year = ds_prev_loop_year_Dec.merge(ds_current_loop_year_Jan_Feb)
		ds_mean_DJF = ds_DJF_incl_dec_from_prev_year.groupby('time.season').mean()

		# add the DJF to then ‘MAM’, ‘JJA’ and ‘SON’
		ds_seasonal_mean_incl_correct_DJF = xr.concat([ds_mean_DJF, ds_mean_grouped_by_season_excluding_DJF], dim='season')

		# add the year coordinate (for DJF, the labelled year is the one corresponding to JF)
		ds_seasonal_mean_incl_correct_DJF = ds_seasonal_mean_incl_correct_DJF.assign_coords(year=year)

		# save seasonal means wiht year in filename
		ds_seasonal_mean_incl_correct_DJF.to_netcdf(seasonal_mean_annual_filename)

