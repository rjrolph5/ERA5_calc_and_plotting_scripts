

### This repo contains scripts that analyze large climate reanalysis datasets 
### (ERA5). For example, you can use scripts in this repo to calculate and plot seasonal means and trends of selected 
### climate variables or create a sea ice mask based on user-specified sea ice concentration data.


### Animations of example output (sea ice concentration) are given in the folder 'sea_ice_concentration'

### Example figures have been produced using the scripts below are contained also in this repo, under the directory : ERA5_plots/seasonal_trends


### ERA5 data is from Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth 
### generation of ECMWF atmospheric reanalyses of the global climate . Copernicus Climate Change Service Climate Data Store (CDS), https://cds.climate.copernicus.eu/cdsapp#!/home



### Brief description of scripts contained in this repo:

# To calculate seasonal means of an ERA5 climate variable, use script below
calculate_seasonal_means_climvars.py

# Plot the seasonal means and trends of the ERA5 cliamte variables you calculated in the script called 'calculate_seasonal_means_climvars.py'
plot_seasonal_means_and_seasonal_trends.py

# Create a sea ice mask (boolean) based on a user-defined sea ice concentration threshold
make_sea_ice_mask.py





