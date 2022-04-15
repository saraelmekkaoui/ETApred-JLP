#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from osgeo import gdal, osr
from skimage.graph import route_through_array
from geopy.distance import geodesic, great_circle
import pygeohash as pgh
import base64
import seaborn as sns
import xarray
from shapely.geometry import Point, Polygon
plt.style.use('seaborn-whitegrid')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

## SET PATHS / VALUES

data_path = 'data_path'
ais_file = 'ais_file.csv'
ship_file = 'ship_file.xlsx'
map_path = 'map.tif'
weather_current_data_path = 'weather_current_data_path'
v_10m_wind_data = 'v_10m_wind_data_path'
u_10m_wind_data = 'u_10m_wind_data_path'
sst_data = 'sst_data_path'
swh_data = 'swh_data_path'
mean_wave_period_data = 'mean_wave_period_data_path'
mean_wave_direction_data = 'mean_wave_direction_data_path'

#destination port latitude and longitude
port_latitude=33.123
port_longitude=-8.631

##  AIS dataset variables ##
############################

#ROW_ID (port call or voyage identifier)
#Vessel_Name            
#IMO                      
#MMSI                     
#Date                              
#Max_Draught                                
#Position_Timestamp      
#Latitude               
#Longitude              
#Speed_over_Ground      
#Course_over_Ground     
#True_Heading           
#Rate_of_Turn           
#Navigational_Status    

##  Vessel particulars dataset variables ##
############################
#IMO                    
#GT            
#DWT            
#LOA         
#BEAM                     
#VesselName      
#MMSI                
#VesselTypeB     


# ========================================================================================================
# ========================================================================================================
## STEP 1: Data Loading and exploration
# ========================================================================================================
# ========================================================================================================

print('')
print('STEP 1: DATA LOADING AND EXPLORATION')
print('')

#=====================================================================================
#Load the datasets
#=====================================================================================
print('Loading data ...')

#df_port = pd.read_excel(data_path + port_file)
df_ais = pd.read_csv(data_path + ais_file)
df_ship = pd.read_excel(data_path + ship_file)


#=====================================================================================
#Shape, features and their types
#=====================================================================================

print('AIS dataset shape is:', df_ais.shape)
print('AIS dataset features and types:')
print(df_ais.dtypes)

print()

print('Ship dataset shape is:', df_ship.shape)
print('Ship dataset features and types:')
print(df_ship.dtypes)

print()

#=====================================================================================
#Merge AIS and ship particulars:
#=====================================================================================
print('Merging AIS and Ship perticulars datasets ...')

df = df_ais.merge(df_ship.drop(['MMSI','VesselName'], axis=1), on = 'IMO', how = 'left')

print('The new dataset df has shape:', df.shape)

print('The dataset features and types:')
print(df.dtypes)

print()


#=====================================================================================
#Change 'Date' and 'Position_Timestamp' to date format and remove zoning info from 'Position_Timestamp'
#=====================================================================================

print('Changing \'Date\' and \'Position_Timestamp\' features type to datetime and removing timezone info from \'Position_Timestamp\':' )

df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d %H:%M")
df['Position_Timestamp'] = pd.to_datetime(df['Position_Timestamp'])
df['Position_Timestamp'] = df['Position_Timestamp'].dt.tz_convert(None)

print('\'Date\' type is:', df.Date.dtype.name)
print('\'Position_Timestamp\' type is:', df.Position_Timestamp.dtype.name)


#=====================================================================================
#Check missing values
#=====================================================================================
print('Checking number of NaNs per column ...')
print(df.isna().sum())
print()

#=====================================================================================
#Providing Information about the dataset
#=====================================================================================

#Statistis =========================================
print('Checking the dataset statistics and detecting data issues:')
display(df.describe())

#Map ===============================================
plt.figure(figsize = (20,10))
m = Basemap(projection='merc',llcrnrlon=-95,llcrnrlat=-25,urcrnrlon=140, urcrnrlat=75, lat_ts=0, resolution='l')
#m.shadedrelief()
m.fillcontinents(color='grey')
#m.drawmapboundary(fill_color='grey')
mx, my = m(df['Longitude'].values, df['Latitude'].values)
#m.plot(mx, my, 'ro', markersize=3)
m.scatter(mx,my, c='red', s=3)
#m.drawcountries(linewidth = 0.5)
#m.drawstates(linewidth = 0.2)
#m.drawcoastlines(linewidth=1)
#plt.tight_layout()
plt.show()

# ========================================================================================================
# ========================================================================================================
## STEP 2: Prepare new variables
# ========================================================================================================
# ========================================================================================================

print('')
print('STEP 2: PREPARATION OF NEW VARIABLES')
print('')
#=====================================================================================
#Calculate the remaining distance to destination port from vessel's position
#=====================================================================================
#Code adapted from https://github.com/tsunghao-huang/Python-Ports-Distance-Calculator

print('Calculate remaining distance to destination port from vessel\'s position ...')

#Transforming a raster map to array datatype.
raster = gdal.Open(map_path)
band = raster.GetRasterBand(1)
mapArray = band.ReadAsArray()

#Get geotransform information and declare some variables for later use
geotransform = raster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3] 
pixelWidth = geotransform[1] 
pixelHeight = geotransform[5]

#transform the coordinates to the exact position in the array.
def coord2pixelOffset(x,y):
    
    xOffset = int((x - originX)/pixelWidth)
    yOffset = int((y - originY)/pixelHeight)
    return xOffset,yOffset

#create a path which travels through the cost map.
def createPath(costSurfaceArray,startCoord,stopCoord):   

    # coordinates to array index
    startCoordX = startCoord[0]
    startCoordY = startCoord[1]
    startIndexX,startIndexY = coord2pixelOffset(startCoordX,startCoordY)

    stopCoordX = stopCoord[0]
    stopCoordY = stopCoord[1]
    stopIndexX,stopIndexY = coord2pixelOffset(stopCoordX,stopCoordY)

    # create path
    indices, weight = route_through_array(costSurfaceArray, (startIndexY,startIndexX), (stopIndexY,stopIndexX),geometric=True,fully_connected=True)
    indices = np.array(indices).T
    indices = indices.astype(float)
    indices[1] = indices[1]*pixelWidth + originX
    indices[0] = indices[0]*pixelHeight + originY
    return indices

#Calculate the vincenty distance starts from the first pair of points to the last.
def calculateDistance(pathIndices):
    distance = 0
    for i in range(0,(len(pathIndices[0])-1)):
        distance += geodesic((pathIndices[0,i], pathIndices[1,i]), (pathIndices[0,i+1], pathIndices[1,i+1])).miles*0.868976
    return distance

#Calculate the distance
def distanceCalculator(startCoord, stopCoord):
    
    pathIndices = createPath(mapArray,startCoord,stopCoord)
    distance = calculateDistance(pathIndices)
    
    return distance


df['remaining_distance'] = df.apply(lambda x: distanceCalculator((x['Longitude'],x['Latitude']), 
                                                                 (port_longitude, port_latitude)), axis=1)

print('Remaining distance from any point calculated')

#=====================================================================================
# Create other features and the Target
#=====================================================================================

# Create the following features
# Distance (circular) between two consecutive trajectory points in nautical miles
# Speed of the vessel: distance per time (nautical miles / hour)
# Elapsed time from the previous to the current trajectory points (Timedelta)

#Sort values by TimeStamp

print('Sorting rows by Timestamp ...')
print('')

df = df.sort_values(by=['Position_Timestamp'])
df = df.reset_index(drop=True)

# Define a list of port calls (voyages)
call_id = df.ROW_ID.unique()

print('Generating new variables and the target ...')
print('')

# Define a function to create all the variables 
def features(dataset, identifier):
# Iterate to compute the new variables listed above.
    prev_row, mask, distance, speed, acc_dist, acc_time, elapsed_time = None, dataset['ROW_ID'] == identifier, [0], 
    [0], [0], [0], [pd.Timedelta('0 days')]
    
    # encode the geo location
    for idx,row in dataset[mask].iterrows():
        if prev_row is None:
            prev_row = row
            continue    
    # compute the distance between the previous checkpoint and the current one
        distance_per_segment = geodesic((row.Latitude, row.Longitude), (prev_row.Latitude, prev_row.Longitude)).miles*0.868976
        distance.append(distance_per_segment)
    # compute the elapsed time between checkpoints
        elapsed_time_per_segment = row.Position_Timestamp - prev_row.Position_Timestamp     
        elapsed_time.append(elapsed_time_per_segment)
    # compute the efficiency, based on the traversed distance per time    
        if elapsed_time_per_segment.total_seconds() > 0:        
            speed.append(distance_per_segment/(elapsed_time_per_segment/np.timedelta64(1, 'h')))
        else:
            speed.append(0.0)
        prev_row = row    

    dataset.loc[mask,'leg_distance'] = distance
    dataset.loc[mask,'leg_speed'] = speed
    dataset.loc[mask,'leg_elapsed_time_Timedelta'] = elapsed_time

#Genreate the variables
for identifier in call_id:
    features(df, identifier)
    
# Get trajectory origin point latitude and longitude
df['Origin_Lat'] = 0
df['Origin_Lon'] = 0
voyages = df.ROW_ID.unique() 
for i in range(len(voyages)):
    v=voyages[i]
    lat = df[df['ROW_ID'] == v].iloc[0,9]
    lon = df[df['ROW_ID'] == v].iloc[0,10]
    df.loc[df['ROW_ID'] == v,'Origin_Lat'] = lat
    df.loc[df['ROW_ID'] == v,'Origin_Lon'] = lon
    
#Get accumulated time and distance from the origin for each voyage

df['acc_dist'] = 0
df['acc_time'] = 0
voyages = df.ROW_ID.unique() 
for i in range(len(voyages)):
    v=voyages[i]
    df.loc[df['ROW_ID'] == v,'acc_dist'] = df.loc[df['ROW_ID'] == v,'leg_distance'].cumsum()
    df.loc[df['ROW_ID'] == v,'acc_time'] = df.loc[df['ROW_ID'] == v,'leg_elapsed_time_Timedelta'].cumsum()

df['acc_time'] = pd.to_timedelta(df['acc_time'])
df['acc_time_hours'] = df.apply(lambda row: row.acc_time / np.timedelta64(1, 'h'), axis=1)

#Leg elapsed time in hours
df['leg_elapsed_time_hours'] = df.apply(lambda row: row.leg_elapsed_time_Timedelta / np.timedelta64(1, 'h'), axis=1)

#Generating vessels' age
df['Age'] = df.apply(lambda row: row.Date.year - row.BuiltYear, axis = 1)

#Generating the target  variable
df['Target'] = (df['Date'] - df['Position_Timestamp']) / np.timedelta64(1, 'h')

print ('''The following variables have been gerenrated:
Leg distance
Leg speed
Leg elapsed time
Trajectory origin point latitude
Trajectory origin point longitude
Accumulated distance from trajectory origin
Accumulated time from trajectory origin
Vessels' age
Target (travel time from the position to JL port in hours)
''')

# ========================================================================================================
# ========================================================================================================
## STEP 3: Data cleaning
# ========================================================================================================
# ========================================================================================================

print('')
print('STEP 3: DATA CLEANING')
print('')

#=====================================================================================
# Outliers detection
#=====================================================================================

print('Plotting variables to inspect visually if there are any outliers ... ')

sns.set(rc={'figure.figsize':(15,10)}, style='white')
fig, axs = plt.subplots(nrows=3, ncols=2)
sns.scatterplot(data=df, x="remaining_distance", y="Target",ax=axs[0,0])
sns.scatterplot(data=df, x=df.index, y="Speed_over_Ground",ax=axs[0,1])
sns.scatterplot(data=df, x=df.index, y="Course_over_Ground",ax=axs[1,0])
sns.scatterplot(data=df, x=df.index, y="Navigational_Status",ax=axs[1,1])
sns.scatterplot(data=df, x=df.index, y="Rate_of_Turn",ax=axs[2,0])
sns.scatterplot(data=df, x=df.index, y="Max_Draught",ax=axs[2,1])
plt.show()
plt.savefig('outliers_plot')

print('')
print('Plotting the target per remaining distance and voyages')

sns.set(rc={'figure.figsize':(15,10)}, style='white')
sns.scatterplot(data=df, x="remaining_distance", y="Target", hue="ROW_ID", legend=False, palette="deep")
plt.show()
plt.savefig('spatial_outliers_plot')

#=====================================================================================
# Remove outliers and fill missing values
#=====================================================================================

# Spatial outliers detection and removal
#================================================

print('Dataset shape before removing outliers:', df.shape)
print('')

print('Removing spatial outliers ...')

#Filter meassages with target below 5 hours and remaining distance above 200 nautical miles
#Filter meassages with target below 60 hours and remaining distance above 2000 nautical miles
target1 = 5
target2 = 60
rem_dist1 = 200
rem_dist2 = 2000

outliers1 = df[(df['Target'] < target1) & (df['remaining_distance'] > rem_dist1)]
out1_list = outliers1.ROW_ID.unique().tolist()
outliers2 = df[(df['Target'] < target2) & (df['remaining_distance'] > rem_dist2)]
out2_list = outliers2.ROW_ID.unique().tolist()
outliers_list = list(set(out1_list + out2_list))


spatial_outliers = df[df['ROW_ID'].isin(outliers_list)]

print('Number of outliers meassages is:', spatial_outliers.shape[0])
print('')

df.drop(spatial_outliers.index, inplace=True)

print('Dataset shape after spatial outliers removal:', df.shape)


print('Dataset missing values:')
print('')

print(df.isna().sum())

print('')

print('Filling missing values and fixing outliers in SOG and draught ...')

# Speed over Ground
#================================================

df.loc[df['Speed_over_Ground'] > 20, 'Speed_over_Ground'] = None
df['Speed_over_Ground'].fillna((df['Speed_over_Ground'].mean()), inplace=True)

# Course over Ground
#================================================

list_cog = df[df['Course_over_Ground'].isna()].ROW_ID.unique()
for i in list_cog:
    mask_cog = df['ROW_ID'] == i
    cog_mean = df.loc[mask_cog, 'Course_over_Ground'].mean()
    df.loc[mask_cog, 'Course_over_Ground'] = df.loc[mask_cog, 'Course_over_Ground'].fillna(cog_mean)
    
# True heading
#================================================
    
mask_th = df['True_Heading'].isna()
df.loc[mask_th,'True_Heading'] = df.loc[mask_th,'Course_over_Ground']


# Rate of Turn 
#================================================
#A rate of turn of 0° will be used to fill NaN values

df['Rate_of_Turn'].fillna(0, inplace=True)

# Navigational Status
#================================================
# Use the most frequent category 0 to fill NaN values

df['Navigational_Status'].fillna(0, inplace=True)


# Draught
#================================================

df.loc[df['Max_Draught'] >= 15, 'Max_Draught'] = None
draught_mean = df['Max_Draught'].mean()
df['Max_Draught'].fillna(draught_mean, inplace=True)



print('Filling missing values completed.')
print('')
print('Dataset missing values:')
print('')
print(df.isna().sum())
print('')

# ========================================================================================================
# ========================================================================================================
## STEP 4: Variables transformation
# ========================================================================================================
# ========================================================================================================

print('')
print('STEP 4: VARIABLES TRANSFORMATION')

#Transforming features with circular nature (Heading, COG, and Rate of Turn) to two trigonometric variables each using sine and cosine functions.')
print('Transforming circular variables ...')
print('')
df['COG_cos'] = df.apply(lambda row: np.cos(row.Course_over_Ground * np.pi / 180.), axis=1) 
df['COG_sin'] = df.apply(lambda row: np.sin(row.Course_over_Ground * np.pi / 180.), axis=1) 
df['TH_cos'] = df.apply(lambda row: np.cos(row.True_Heading * np.pi / 180.), axis=1) 
df['TH_sin'] = df.apply(lambda row: np.cos(row.True_Heading * np.pi / 180.), axis=1) 
df['ROT_cos'] = df.apply(lambda row: np.cos(row.Rate_of_Turn * np.pi / 180.), axis=1) 
df['ROT_sin'] = df.apply(lambda row: np.cos(row.Rate_of_Turn * np.pi / 180.), axis=1) 

print('One-hot encoding of categorical variables ...')
print('')

Navigational_Status  = pd.get_dummies(df.Navigational_Status, prefix='Navigational_Status')
VesselTypeB  = pd.get_dummies(df.VesselTypeB, prefix='VesselTypeB')
df = pd.concat([df, Navigational_Status, VesselTypeB], axis=1)

print('Trasformation done.')

# ========================================================================================================
# ========================================================================================================
## STEP 5: Weather data integration
# ========================================================================================================
# ========================================================================================================

print('')
print('STEP 5: WEATHER DATA INTEGRATION')
print('')

#use this function to . 
#provide the weather data path, 
def integrate_weather(data_path, weather_data_type, variable):
    
    """
    Integrates weather data with the dataset
    
    Arguments:
    data_path -- weather data path
    weather_data_type -- the name of the weather variables to display as the column name in the dataset
    variable -- the specific variable that will be integrated from weather data file 
    
    Returns:
    df -- the dataset df containng the integrated weather variable
    """
    
    print('Reading weather data ...')
    data = xarray.open_mfdataset(data_path)
    print('Printing weather data info:')
    display(data)
    print('Integrating weather data with the dataset ...')
    df['weather_data_type'] = df.apply(lambda row: data.sel(latitude=row.Latitude, longitude=row.Longitude,
                                                            time=row.Position_Timestamp,method="nearest").variable.values, 
                                                            axis=1
return df



#Current data (Oceanic General Circulation)
#===================================

df = integrate_weather(weather_path_current, current_uo, uo)
df = integrate_weather(weather_path_current, current_uo, uo)

#Mean wave direction 
#======================================
                                       
df = integrate_weather(mean_wave_direction_data, mwd, mwd)                                       

#Mean wave period
#======================================

df = integrate_weather(mean_wave_period_data, mwp, mwp) 

#Significant_height_combined_wind_waves_swell (swh)
#======================================
                                       
df = integrate_weather(swh_data, swh, swh)

#Sea surface temperature
#======================================
                                       
df = integrate_weather(sst_data, sst, sst)

#10 m u-component and v-component of wind
#======================================
                
df = integrate_weather(u_10m_wind_data, wind_u10, u10)
df = integrate_weather(v_10m_wind_data, wind_v10, v10)


#Printing df weather data types 
#======================================

print('Printing data types ...')
print('')
print(df.dtypes)


#Changing weather data types 
#======================================

print('Changing weather data type to float ...')
print('')

df['current_uo'] = df['current_uo'].astype(float)
df['current_vo'] = df['current_vo'].astype(float)
df['wind_u10'] = df['wind_u10'].astype(float)
df['wind_v10'] = df['wind_v10'].astype(float)
df['mwd'] = df['mwd'].astype(float)
df['mwp'] = df['mwp'].astype(float)
df['swh'] = df['swh'].astype(float)
df['sst'] = df['sst'].astype(float)

print('')

print('Checking missing values in weather data')
print('')

print(df.isna().sum())

print('Fixing missing values in weather data')
print('')

print('current_uo description:')
df.current_uo.describe()
print('current_vo description:')
df.current_vo.describe()
print('wind_u10 description:')
df.wind_u10.describe()
print('wind_v10 description:')
df.wind_v10.describe()
print('mwd description:')
df.mwd.describe()
print('mwp description:')
df.mwp.describe()
print('swh description:')
df.swh.describe()
print('sst description:')
df.sst.describe()

print('')
print('Filling missing values ...')

df['current_uo'].fillna(0, inplace=True)
df['current_vo'].fillna(0, inplace=True)
df['mwp'].fillna((df['mwp'].mean()), inplace=True)
df['mwd'].fillna((df['mwd'].mean()), inplace=True)
df['swh'].fillna((df['swh'].mean()), inplace=True)
df['sst'].fillna((df['sst'].mean()), inplace=True)

print('Weather data integration is done')

# ========================================================================================================
# ========================================================================================================
## STEP 6: Information about the dataset
# ========================================================================================================
# ========================================================================================================

#Features names and types
#================================================

print('The dataset features and types are:')
print(df.dtypes)
print('')

#Check missing values
#================================================

print('Check if there are any missing values:')
print(df.isna().sum())
print('')

#Distribution and statistics of the route legs time (delta_time)
#================================================


print('Information about the delta time between positions:')
sns.distplot(df['leg_elapsed_time_hours'])
print(df.leg_elapsed_time_hours.describe())

