

# Plot seasonal means & trends of seasonal means of user-specified variables from ERA5 data.
# Seasonal means have been calculated using the script: calculate_seasonal_means_climvars.py

from netCDF4 import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs # cartopy coordinate reference system.
from scipy import ndimage
from cartopy.util import add_cyclic_point
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from pathlib import Path
import cmocean
from matplotlib import cm


########## load a climate variable file

#climvar = '2m_temperature'
#var_shortname = 't2m'

#climvar = 'sea_ice_area_fraction'
#var_shortname = 'siconc'

climvar = 'era5_wind_speed'
var_shortname = 'wind_speed_10m'

########## choose if you want to calculate seasonal trends (e.g. if they have not been calculated already)
calc_seasonal_trends_bool = False

########## choose if you want to plot seasonal trends
plot_seasonal_trend_bool = True # if true, will plot the seasonal trends.
plot_seasonal_means_bool = False # if true, it will plot seasonal means (which have been calculated in a separate script called ../calculate_seasonal_means_climvars.py)

########## define datapaths
datapath = '/permarisk/data/ERA_Data/ERA5_becca/'
trends_and_regressions_basepath = datapath + 'era5_seasonal_means/trends_and_regressions/' + climvar + '/'
Path(trends_and_regressions_basepath).mkdir(parents=True, exist_ok=True)
plot_path = '/permarisk/output/ERA5_plots/'

########## calculate the seasonal trends
# pre-process the data so that the year is included as a dimension and not just a coord
def preprocessing(ds):
	return ds.expand_dims(dim='year')

########## Calculate and save the trends and regression for each season into separate files
def find_and_save_seasonal_trend_at_each_pixel(var_shortname, ds_seasonal_data, season): # season is str (e.g. 'DJF')

	# extract the seasonal means from ds above into yearly timeseries, one timeseries for each season
	annual_seasonal_data = ds_seasonal_data[var_shortname].sel(season=season)

	vals = annual_seasonal_data.values # (years, latitude, longitude) ... aka ... (time, y, x)
	years = annual_seasonal_data.year.values

	if season == 'DJF':
		vals = vals[1:,:,:] # removes the first year of the year dimension, 1979, when calculating DJF trends
		years = annual_seasonal_data.year.values[1:]

	# reshape the array with as many rows as years and as many columns as pixels.
	vals2 = vals.reshape(len(years), -1)
	# do a first-degree polyfit ### this is where the trend is calculated (via polyfit)
	regressions = np.polyfit(years, vals2, 1)
	# get the coefficients back
	trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
	y_int = regressions[1,:].reshape(vals.shape[1], vals.shape[2])

	# You can visualise the trends output variable to see your trend map. You should also check out the
	# goodness-of-fit using the second row of outputs in regressions.

	return (regressions, trends)


########## plot contour figure of seasonal trends

# load the pre-calculated seasonal trends
if calc_seasonal_trends_bool == True:
	### use dask to open files all at once, as read-only.
	ds_seasonal_data = xr.open_mfdataset(datapath + 'era5_seasonal_means/' + var_shortname + '/seasonal_mean_*', combine = 'by_coords', concat_dim = 'year', preprocess = preprocessing) # combine= 'nested' or 'by_coords' see http://xarray.pydata.org/en/stable/combining.html#combining-multi
	for season in ['DJF','MAM', 'JJA', 'SON']:
		regressions, trends = find_and_save_seasonal_trend_at_each_pixel(var_shortname, ds_seasonal_data, season)
		# save the regression to file
		np.save(trends_and_regressions_basepath + 'regressions_' + season + '_' + var_shortname + '.npy', regressions)
		# save the trends file
		np.save(trends_and_regressions_basepath + 'trends_' + season + '_' + var_shortname + '.npy', trends)

def find_min_and_max_for_colorbar_trends_and_regressions():
	# initialize the trend and regressions array
	trends_all_min_values = np.array([1])
	trends_all_max_values = np.array([1])
	regressions_all_min_values = np.array([1])
	regressions_all_max_values = np.array([1])
	for season in ['DJF', 'MAM', 'JJA', 'SON']:
		## find min and max for colorbars of the seasonal trends
		# load the regressions and trends that were calculated above
		regressions = np.load(trends_and_regressions_basepath + 'regressions_' + season + '_' + var_shortname + '.npy')
		trends = np.load(trends_and_regressions_basepath + 'trends_' + season + '_' + var_shortname + '.npy')
		# find the min and max
		regressons_min = np.nanmin(regressions)
		trends_min = np.nanmin(trends)
		trends_max = np.nanmax(trends)
		regressions_max = np.nanmax(regressions)
		regressions_min = np.nanmin(regressions)
		# append to overall array across all seasons to find min max that works for all 4.
		trends_all_min_values = np.append(trends_all_min_values, trends_min)
		trends_all_max_values = np.append(trends_all_max_values, trends_max)
		regressions_all_min_values = np.append(regressions_all_min_values, regressions_min)
		regressions_all_max_values = np.append(regressions_all_max_values, regressions_max)
	# remove initialization value
	trends_all_min_values = trends_all_min_values[1:]
	trends_all_max_values = trends_all_max_values[1:]
	regressions_all_min_values = regressions_all_min_values[1:]
	regressions_all_max_values = regressions_all_max_values[1:]
	# find the min across all seasons
	trends_overall_min = np.nanmin(trends_all_min_values)
	regressions_overall_min = np.nanmin(regressions_all_min_values)
	# find the max across all seasons
	trends_overall_max = np.nanmax(trends_all_max_values)
	regressions_overall_max = np.nanmax(regressions_all_max_values)

	return(trends_overall_min, trends_overall_max, regressions_overall_min, regressions_overall_max)


# define a plotting function for the contour trends
def plot_contour_trends(data_to_plot, plot_filename, units_str, plot_title_str, step_size_for_contour, coefficients_to_convert_data_to_above_units, step_size_for_ticks):

	fig = plt.figure(figsize=(5,5))
	map_proj = ccrs.NorthPolarStereo(central_longitude=0.0, true_scale_latitude=75)
	ax = plt.axes(projection=map_proj)
	ax.coastlines()

	# open an example xarray ds to get the lat lon info
	seasonal_mean_ds = xr.open_dataset(datapath + 'era5_seasonal_means/siconc/seasonal_mean_dec1990JF_MAM_JJA_SON_1991_siconc.nc')

	# add cyclic point to avoid blank space in the longitude array
	data_to_plot = add_cyclic_point(data_to_plot) #, dim='longitude')
	lon_array = np.append(seasonal_mean_ds.longitude.data,180)

	#cmap = plt.cm.coolwarm
	cmap = plt.set_cmap('bwr')

	# load max and min values for colorbar
	min_of_all_files = trends_overall_min*coefficients_to_convert_data_to_above_units
	max_of_all_files = trends_overall_max*coefficients_to_convert_data_to_above_units

	# set levels for values for colorbar contours
	levels_rhs = np.arange(0, max_of_all_files+step_size_for_contour, step_size_for_contour)
	levels_lhs_reversed = np.arange(0, -min_of_all_files, step_size_for_contour)
	levels_lhs = -np.flip(levels_lhs_reversed)[:-1]
	level_values = np.append(levels_lhs,levels_rhs)
	levels = level_values # keeping same wording for contourf

	# set levels for values for ticks on colorbar
	ticks_rhs = np.arange(0, max_of_all_files+step_size_for_contour, step_size_for_ticks)
	ticks_lhs_reversed = np.arange(0, -min_of_all_files, step_size_for_ticks)
	ticks_lhs = -np.flip(ticks_lhs_reversed)[:-1]
	tick_values = np.append(ticks_lhs,ticks_rhs)

	# set 0 as white
	norm = colors.DivergingNorm(vmin=min_of_all_files, vcenter=0, vmax=max_of_all_files)

	c = ax.contourf(lon_array, seasonal_mean_ds.latitude.data, data_to_plot*coefficients_to_convert_data_to_above_units, levels=levels, vmin=min_of_all_files, vmax=max_of_all_files, norm=norm, transform=ccrs.PlateCarree(),cmap=cmap)
	fmt = ticker.ScalarFormatter(useMathText=True)

	cbar = plt.colorbar(c, format=fmt, ticks=tick_values, spacing='uniform')
	cbar.ax.set_ylabel(units_str, size = 'large', weight='bold')

	plt.title(plot_title_str)
	plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close(fig)

	return

# call the seasonal trend plotting functions
if plot_seasonal_trend_bool == True:
	# pull mins and maxs for the colorbar
	trends_overall_min, trends_overall_max, regressions_overall_min, regressions_overall_max = find_min_and_max_for_colorbar_trends_and_regressions()
	# set units
	if climvar == '2m_temperature':
		plot_title_strbase = ' 1979-2020 Seasonal Average Trend: \n 2m temp'
		degree_sign= u'\N{DEGREE SIGN}'
		units_str = '[' + degree_sign + 'C / decade]'
		step_size_for_contour = 0.1 # manually select from data plotted
		coefficients_to_convert_data_to_above_units = 10
		step_size_for_ticks = 5*step_size_for_contour

	if climvar == 'sea_ice_area_fraction':
		plot_title_strbase = ' 1979-2020 Seasonal Average Trend: \n sea ice concentration'
		units_str = '[% / decade]'
		step_size_for_contour = 0.5
		coefficients_to_convert_data_to_above_units = 100*10
		step_size_for_ticks = 10*step_size_for_contour

	if climvar == 'era5_wind_speed':
		plot_title_strbase = ' 1979-2020 Seasonal Average Trend: \n wind speed'
		units_str = '[m/s per decade]'
		step_size_for_contour = 0.01
		coefficients_to_convert_data_to_above_units = 1 # data is already in the above units.
		step_size_for_ticks = 1*step_size_for_contour
	# run the plotting function for each season
	for season in ['DJF', 'MAM', 'JJA', 'SON']:
		plot_title_str = season + plot_title_strbase
		data_to_plot = np.load(trends_and_regressions_basepath + 'trends_' + season + '_' + var_shortname + '.npy')
		plot_filename = plot_path + 'seasonal_trends/' + season + '/' + season + '_trend_' + climvar + '.png'
		plot_filename = plot_contour_trends(data_to_plot, plot_filename, units_str, plot_title_str, step_size_for_contour, coefficients_to_convert_data_to_above_units, step_size_for_ticks)


################# plot contour figure of seasonal means #########################

def plot_contour_seasonal_means(data_to_plot, plot_filename, units_str, plot_title_str, step_size_for_contour, value_to_convert_data_to_above_units, step_size_for_ticks, min_for_colorbar, max_for_colorbar):

	fig = plt.figure(figsize=(5,5))
	map_proj = ccrs.NorthPolarStereo(central_longitude=0.0, true_scale_latitude=75)
	ax = plt.axes(projection=map_proj)
	ax.coastlines()

	# open an example xarray ds to get the lat lon info
	seasonal_mean_ds = xr.open_dataset(datapath + 'era5_seasonal_means/siconc/seasonal_mean_dec1990JF_MAM_JJA_SON_1991_siconc.nc')

	# add cyclic point to avoid blank space in the longitude array
	data_to_plot = add_cyclic_point(data_to_plot) #, dim='longitude')
	lon_array = np.append(seasonal_mean_ds.longitude.data,180)

	#cmap = plt.cm.coolwarm
	cmap = plt.set_cmap('bwr')

	# load max and min values for colorbar
	min_of_all_files = min_for_colorbar + value_to_convert_data_to_above_units
	max_of_all_files = max_for_colorbar + value_to_convert_data_to_above_units

	# set levels for values for colorbar contours
	levels_rhs = np.arange(0, max_of_all_files+step_size_for_contour, step_size_for_contour)
	levels_lhs_reversed = np.arange(0, -min_of_all_files, step_size_for_contour)
	levels_lhs = -np.flip(levels_lhs_reversed)[:-1]
	level_values = np.append(levels_lhs,levels_rhs)
	levels = level_values # keeping same wording for contourf

	# set levels for values for ticks on colorbar
	ticks_rhs = np.arange(0, max_of_all_files+step_size_for_contour, step_size_for_ticks)
	ticks_lhs_reversed = np.arange(0, -min_of_all_files, step_size_for_ticks)
	ticks_lhs = -np.flip(ticks_lhs_reversed)[:-1]
	tick_values = np.append(ticks_lhs,ticks_rhs)

	# set different colors for colorbars depending on the dataset
	if var_shortname == 'wind_speed_10m':
		cmap = plt.get_cmap('Reds')
		c = ax.contourf(lon_array, seasonal_mean_ds.latitude.data, data_to_plot + value_to_convert_data_to_above_units, levels=levels, vmin=min_of_all_files, vmax=max_of_all_files, transform=ccrs.PlateCarree(),cmap=cmap)

	elif var_shortname == 'siconc':
		cmap = cmocean.cm.ice
		c = ax.contourf(lon_array, seasonal_mean_ds.latitude.data, data_to_plot + value_to_convert_data_to_above_units, levels=levels, vmin=min_of_all_files, vmax=max_of_all_files, transform=ccrs.PlateCarree(),cmap=cmap)
	else:
		norm = colors.DivergingNorm(vmin=min_of_all_files, vcenter=0, vmax=max_of_all_files)
		c = ax.contourf(lon_array, seasonal_mean_ds.latitude.data, data_to_plot + value_to_convert_data_to_above_units, levels=levels, vmin=min_of_all_files, vmax=max_of_all_files, norm=norm, transform=ccrs.PlateCarree(),cmap=cmap)

	fmt = ticker.ScalarFormatter(useMathText=True)

	cbar = plt.colorbar(c, format=fmt, ticks=tick_values, spacing='uniform')
	cbar.ax.set_ylabel(units_str, size = 'large', weight='bold')

	plt.title(plot_title_str)
	plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close(fig)
	return

##### define args for plotting seasonal mean function

ds_seasonal_data = xr.open_mfdataset(datapath + 'era5_seasonal_means/' + var_shortname + '/seasonal_mean_*', combine = 'by_coords', concat_dim = 'year', preprocess = preprocessing) # combine= 'nested' or 'by_coords' see http://xarray.pydata.org/en/stable/combining.html#combining-multi

vals = ds_seasonal_data[var_shortname].values
min_for_colorbar = np.nanmin(vals)
max_for_colorbar = np.nanmax(vals)


## set units, plot title, coefficients for unit conversion, etc.
if climvar == '2m_temperature':
	degree_sign= u'\N{DEGREE SIGN}'
	units_str = '[' + degree_sign + 'C]'
	step_size_for_contour = 5 # manually select from data plotted
	value_to_convert_data_to_above_units = -273.15 # this is added to the data from ERA5
	step_size_for_ticks = 1*step_size_for_contour

if climvar == 'sea_ice_area_fraction':
	units_str = '[fraction]'
	step_size_for_contour = 0.02
	value_to_convert_data_to_above_units = 0
	step_size_for_ticks = 5*step_size_for_contour

if climvar == 'era5_wind_speed':
	units_str = '[m/s]'
	step_size_for_contour = 1
	value_to_convert_data_to_above_units = 0 # this is added , so adding 0 means data is already in the above units.
	step_size_for_ticks = 1*step_size_for_contour

if plot_seasonal_means == True:
	for year in np.arange(1979,2021):
		print(year)
		## data_to_plot
		if year != 1979:
	        	ds_seasonal_data_ifile = datapath + 'era5_seasonal_means/' + var_shortname + '/seasonal_mean_dec' + str(year-1) + 'JF_MAM_JJA_SON_' + str(year) + '_' + var_shortname + '.nc'
		if year == 1979:
	        	ds_seasonal_data_ifile = datapath + 'era5_seasonal_means/' + var_shortname + '/seasonal_mean_MAM_JJA_SON_1979_' + var_shortname + '.nc'
		for season in ['DJF', 'MAM', 'JJA', 'SON']:
			print(season)
			## plot_filename
			# plot_path has the year with most of the months in it (e.g. year 1980 has 1 month dec in the preceding year, 1979)
			if season == 'DJF' and year == 1979:
				print('DJF is not available for 1979')
			else:
				plot_path_seasonal_mean = plot_path + 'seasonal_means/' + var_shortname + '/' + season + '/'
				Path(plot_path_seasonal_mean).mkdir(parents=True, exist_ok=True)
				plot_filename = plot_path_seasonal_mean + var_shortname + '_seasonal_mean_' + season + '_' + str(year) + '.png'

				seasonal_mean_ds = xr.open_dataset(ds_seasonal_data_ifile)
				annual_seasonal_data = seasonal_mean_ds[var_shortname].sel(season=season)
				data_to_plot = annual_seasonal_data.values # (years, latitude, longitude) ... aka ... (time, y, x)
				plot_title_str = climvar + ' ' + season + ' ' + str(year)

				plot_contour_seasonal_means(data_to_plot, plot_filename, units_str, plot_title_str, step_size_for_contour, value_to_convert_data_to_above_units, step_size_for_ticks, min_for_colorbar, max_for_colorbar)

