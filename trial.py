import xarray as xr
from scipy.interpolate import griddata
import numpy as np

# Load Data 1 and Data 2
data1 = xr.open_dataset('data1.nc')
data2 = xr.open_dataset('data2.nc')

# Extract relevant variables
lon1 = data1['lon']  # (i, j)
lat1 = data1['lat']  # (i, j)
thick1 = data1['thick']  # (time, i, j)

lon2 = data2['lon']  # (lon)
lat2 = data2['lat']  # (lat)
thick2 = data2['thick']  # (time, lat, lon)

# Flatten the grid of Data 1
points = np.array([lon1.values.flatten(), lat1.values.flatten()]).T
thick1_flat = thick1.values.reshape(thick1.shape[0], -1)

# Create the target grid (from Data 2)
lon2_grid, lat2_grid = np.meshgrid(lon2, lat2)
target_points = np.array([lon2_grid.flatten(), lat2_grid.flatten()]).T

# Interpolate thick1 onto the grid of Data 2
thick1_interpolated = np.array([
    griddata(points, thick1_flat[t], target_points, method='linear').reshape(lon2_grid.shape)
    for t in range(thick1.shape[0])
])

# Convert to xarray DataArray
thick1_interp_da = xr.DataArray(
    thick1_interpolated,
    dims=('time', 'lat', 'lon'),
    coords={'time': thick1['time'], 'lat': lat2, 'lon': lon2}
)

# Calculate the difference
difference = thick1_interp_da - thick2

# Save the result
difference.to_netcdf('difference.nc')
