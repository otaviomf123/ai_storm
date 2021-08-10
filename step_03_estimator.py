import sys
### Domain configuration 
lat_min=-32
lat_max=-22
lon_max=-45
lon_min=-60
dx_dy=0.05
### Output savedir 
outdir='.'
## batch size and the size of each batch will have, 
## for example, estimates of 256 will be made and each time 
## the information goes through the models, 10 estimates of 256 will be made
b_size=256
b1_size=int(256*30)
n = len(sys.argv)
lista= []
for n in range(1,n):
    lista.append(str(sys.argv[n]))
print(lista)

import numpy as np
import datetime as dt
import glob
import json
from tqdm import trange
import h5py
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LeakyReLU
from sklearn.neighbors import BallTree

## 
## If you are going to use GPU 
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

## creating grid points
## will be updated in the next version with improvements in the representation of the points 
lat_grid=np.arange(lat_min,lat_max,dx_dy)
lon_grid=np.arange(lon_min,lon_max,dx_dy)
lon_grid,lat_grid=np.meshgrid(lon_grid,lat_grid)
print('total point', lon_grid.shape[0]*lon_grid.shape[1],'shape lon/lat',lon_grid.shape)
print('total interations ',int((lon_grid.shape[0]*lon_grid.shape[1])/b1_size))

# load json and create model for multlevel ref 
json_file = open('preload_files/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,custom_objects={
        'LeakyReLU': LeakyReLU},)
# load weights into new model
model.load_weights("preload_files/model_v02.h5")
print("Loaded model from disk")
# load json and create model
json_file = open('preload_files/model_class.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_class = model_from_json(loaded_model_json,custom_objects={
        'LeakyReLU': LeakyReLU},)
# load weights into new model
model_class.load_weights("preload_files/model_class.h5")
print("Loaded model from disk")

max_val=[325.,325.,325.,325.,325.,325.,325.,325.,30.,1.]
min_val=[183.,183.,183.,183.,183.,183.,183.,183.,-30.,0]
scaler=MinMaxScaler((0,1)).fit([max_val,min_val])

## Loading files with database models, distance from satellite 
bt = pickle.load(open('preload_files/proc_sat_dataset.dat', 'rb'))
train_idxs=np.load('preload_files/train_idxs.npy')

#Calculating index of the closest raw data nearest satellite
request_points=[]
for i in range(len(lon_grid)):
    for j in range(len(lon_grid[i])):
        request_points.append([lat_grid[i,j],lon_grid[i,j]])
request_points=np.array(request_points)
distance,index=bt.query(np.deg2rad(request_points))

for filename in lista:
    path='%s/model_REFHIDRO_%s.hdf5'%(outdir,filename[-21:-5])
    print('creat',path,'from',filename)
    base_h5=h5py.File(filename,'r')
    lat=base_h5['lat'][:]
    lon=base_h5['lon'][:]
    data_goes=base_h5['data'][:]
    raio=base_h5['GLM'][:]
    base_h5.close()
    ####
    dx_goes=16
    goes_dt=[]
    latp=[]
    lonp=[]
    pred=[]
    pred_class=[]
    goes_bruto=[]
    goes_bruto_dt=[]
    for i in trange(len(index)):
        idx=index[i,0]
        i,j=train_idxs[idx]
        data_loop=[]
        gb=[]
        for ch in range(1,len(data_goes)):
            data_loop.append(data_goes[ch,i-dx_goes:i+(dx_goes+1),j-dx_goes:j+(dx_goes+1)])
            gb.append(data_goes[ch,i,j])
        gb.append(data_goes[-2,i,j]-data_goes[-4,i,j])
        gb.append(raio[i,j])
        data_loop.append(data_goes[-2,i-dx_goes:i+(dx_goes+1),j-dx_goes:j+(dx_goes+1)]-data_goes[-4,i-dx_goes:i+(dx_goes+1),j-dx_goes:j+(dx_goes+1)])
        data_loop.append(raio[i-dx_goes:i+(dx_goes+1),j-dx_goes:j+(dx_goes+1)])
        goes_dt.append(data_loop)
        goes_bruto_dt.append(gb)
        if len(goes_dt)==b1_size or idx==index[-1]:
            goes_dt=np.array(goes_dt)
            print(goes_dt.shape)
            goes_dt=np.moveaxis(goes_dt,1,-1)
            ################
            goes_bruto_dt=np.array(goes_bruto_dt)
            goes_bruto_dt=np.moveaxis(goes_bruto_dt,1,-1)
            ############
            sh=goes_dt.shape
            goes_dt=np.reshape(scaler.transform(np.reshape(goes_dt,(sh[0]*sh[1]*sh[2],sh[3]))),sh) 
            print('Estimate reflectivity')
            p=np.array(model.predict(goes_dt,batch_size=b_size,verbose=1))*10
            print('Estimate hydrometeors categories')
            pcl=np.array(model_class.predict(goes_dt,batch_size=b_size,verbose=1))
            for ji in range(len(p)):
                pred.append(p[ji])
                pred_class.append(pcl[ji])
                goes_bruto.append(goes_bruto_dt[ji])
            goes_dt=[]
            goes_bruto_dt=[]

    pred=np.array(pred)
    pred_class=np.array(pred_class)
    goes_bruto=np.array(goes_bruto)
    h = h5py.File(path, 'w')
    dset = h.create_dataset('ref_02', data=np.reshape(pred[:,0],lon_grid.shape))
    dset = h.create_dataset('ref_04', data=np.reshape(pred[:,1],lon_grid.shape))
    dset = h.create_dataset('ref_06', data=np.reshape(pred[:,2],lon_grid.shape))
    dset = h.create_dataset('ref_08', data=np.reshape(pred[:,3],lon_grid.shape))
    dset = h.create_dataset('ref_10', data=np.reshape(pred[:,4],lon_grid.shape))
    dset = h.create_dataset('ref_12', data=np.reshape(pred[:,5],lon_grid.shape))
    dset = h.create_dataset('lat', data=lat_grid)
    dset = h.create_dataset('lon', data=lon_grid)
    dset = h.create_dataset('hidro_hail', data=np.reshape(pred_class[:,0],lon_grid.shape))
    dset = h.create_dataset('hidro_ice', data=np.reshape(pred_class[:,1],lon_grid.shape))
    dset = h.create_dataset('hidro_water', data=np.reshape(pred_class[:,2],lon_grid.shape))
    dset = h.create_dataset('no_hidro', data=np.reshape(pred_class[:,3],lon_grid.shape))
    dset = h.create_dataset('goes', data=np.reshape(goes_bruto,(lon_grid.shape[0],lon_grid.shape[1],10)))
    h.close()
