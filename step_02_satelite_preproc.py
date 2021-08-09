from pathlib import Path
import sys
##change this variable if you need to save the satellite outputs in another folder
outdir='.'
home = str(Path.home())
goes_path='%s/goes16'%home
print('Goes files save dir', goes_path)
####
# check if the number of arguments is correct
n = len(sys.argv)
print("Total arguments passed:", n)
if n!=6:
    raise Exception('argument numbers must be 5 year month day hour minute')
# Arguments passed
print("Year", sys.argv[1])
print("Month", sys.argv[2])
print("Day", sys.argv[3])
print("Hour", sys.argv[4])
print("Minute", sys.argv[5])
year =int(sys.argv[1])
month=int(sys.argv[2])
day=int(sys.argv[3])
hour=int(sys.argv[4])
minute=int(sys.argv[5])
## import the required modules##
from netCDF4 import Dataset
from pyproj import Proj
import numpy as np
import datetime as dt
import glob
import json
import random
from tqdm import trange
import h5py
import os
from sklearn.neighbors import BallTree
import pickle
#### Aux funcs 
## functions that will process the satellite data to prepare it for NN 
def get_filename(ano,mes,dia,hora,minu,canal):
    data=dt.datetime(ano,mes,dia,hora,minu)
    path_1='%s/%04d/%02d/%02d/ABI-L2-CMIPF/%02d/C%02d/'%(goes_path,data.year,
                                                         data.month,
                                                         data.day,
                                                         data.hour,
                                                         canal)
    tt = data.timetuple()
    jd=tt.tm_yday
    try:
        path_2='OR_ABI-L2-CMIPF-M6C%02d_G16_s%04d%03d%02d%01d*'%(canal,data.year,jd,data.hour,data.minute)
        #print(path_1+path_2)
        res=glob.glob(path_1+path_2)[0]
        print('find' , res)
        return res
    except:
        path_2='OR_ABI-L2-CMIPF-M3C%02d_G16_s%04d%03d%02d%01d*'%(canal,data.year,jd,data.hour,data.minute)
        res=glob.glob(path_1+path_2)[0]
        print('find' , res)
        return res

def get_raio_name(ano,mes,dia,hora,minu,inter=1):
    d=dt.datetime(ano,mes,dia,hora,minu)
    tt = d.timetuple()
    jd=tt.tm_yday
    path1='%s/%04d/%02d/%02d/GLM-L2-LCFA/%02d/'%(goes_path,d.year,d.month,d.day,d.hour,)
    path2='OR_GLM-L2-LCFA_G16_s%04d%03d%02d%02d*'%(d.year,jd,d.hour,d.minute)
    #print(path1+path2)
    try:
        
        return glob.glob(path1+path2)
    except:
        return False

def creat_raio_array(date):
    valor=1.
    raio_array=np.zeros(lat.shape[0]*lat.shape[1])
    gr_lon,gr_lat=[],[]
    val=[]
    for tdelta in range(10):
        ddate=date+dt.timedelta(minutes=tdelta)
        raio_name=get_raio_name(ddate.year,ddate.month,ddate.day,ddate.hour,ddate.minute)
        #print(nc_raio.variable)
        if len(raio_name)>0:
            for file in raio_name:
                nc_raio=Dataset(file)
                gr_lo,gr_la=nc_raio['group_lon'][:],nc_raio['group_lat'][:]
                values=np.full(len(gr_lo),valor)
                valor=0.1+(tdelta/10.)
                for idx in range(len(values)):
                    gr_lon.append(gr_lo[idx])
                    gr_lat.append(gr_la[idx])
                    val.append(values[idx])
    gr_lon,gr_lat=np.array(gr_lon),np.array(gr_lat)
    val=np.array(val)
    aj=[]
    a=0
    e=0
    target_points=[]
    for hs in trange(len(gr_lon)):
        chosen_lon=gr_lon[hs]
        chosen_lat=gr_lat[hs]
        target_points.append([chosen_lat,chosen_lon])
        a=a+1
        #print(i,j)
    target_points=np.array(target_points)
    target_points=np.deg2rad(target_points)
    distances, indices = bt.query(target_points)
    for kinx in range(len(indices)):
        raio_array[indices[kinx]]=val[kinx]
    return np.reshape(raio_array,(lat.shape))
def sat_data_creat(ano,mes,dia,hora,minu,outdir):
    data=[]
    for i in range(8,17):
        f_satel=get_filename(ano,mes,dia,hora,minu,i)
        nc = Dataset(f_satel)
        R = nc.variables['CMI'][:]
        data.append(R[cutoff:-cutoff,cutoff:-cutoff])
    path='%s/GOES_16_%04d-%02d-%02d_%02d:%02d.hdf5'%(outdir,ano,mes,dia,hora,minu)
    data=np.array(data)
    raio=creat_raio_array(dt.datetime(ano,mes,dia,hora,minu))
    f=h5py.File(path, "w")
    f.create_dataset("lon", data=lon)
    f.create_dataset("lat", data=lat)
    f.create_dataset("data", data=data)
    f.create_dataset("GLM", data=raio)
    print('file creat complete ',path)
    print('ok',ano,mes,dia,hora,minu)
    
    
####load a file to extract lat lon and index these variables into the geospatial database (bt)
filename=get_filename(year,month,day,hour,0,13)
nc = Dataset(filename)
sat_h = nc.variables['goes_imager_projection'].perspective_point_height
# Satellite longitude
sat_lon = nc.variables['goes_imager_projection'].longitude_of_projection_origin
# Satellite sweep
sat_sweep = nc.variables['goes_imager_projection'].sweep_angle_axis
# the scanning angle (in radians) multiplied by the satellite height (http://proj4.org/projections/geos.html)
X = nc.variables['x'][:] * sat_h
Y = nc.variables['y'][:] * sat_h
R = nc.variables['CMI'][:]
XX, YY = np.meshgrid(X, Y)
p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
lons, lats = p(XX, YY, inverse=True)
lats[np.isnan(R)] = np.nan
lons[np.isnan(R)] = np.nan
lats[np.isinf(lats)] = np.nan
lons[np.isinf(lons)] = np.nan
cutoff=800
lat=lats[cutoff:-cutoff,cutoff:-cutoff]
lon=lons[cutoff:-cutoff,cutoff:-cutoff]
bt = pickle.load(open('preload_files/sat_dataset.pik', 'rb'))
######
print('min-max domain allow lats',np.min(lat),np.max(lat))
print('min-max domain allow lons',np.min(lon),np.max(lon))
##
## future updates will improve converter performance 
## the proper code wastes a lot of time loading static parameters 
sat_data_creat(year,month,day,hour,minute,outdir)

