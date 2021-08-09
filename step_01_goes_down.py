import sys
####
# check if the number of arguments is correct
n = len(sys.argv)
print("Total arguments passed:", n)
if n!=5:
    raise Exception('Argument numbers must be 4 year month day hour')
# Arguments passed
print("Year", sys.argv[1])
print("Month", sys.argv[2])
print("Day", sys.argv[3])
print("Hour", sys.argv[4])
year =int(sys.argv[1])
month=int(sys.argv[2])
day=int(sys.argv[3])
hour=int(sys.argv[4])


from goespy.Downloader import GLM_Downloader
from goespy.Downloader import ABI_Downloader
import datetime as dt

product = 'ABI-L2-CMIPF'
Bucket = 'noaa-goes16' 
## Download parameters for Goes 16 products
d=dt.datetime(year ,month,day,hour)
year ='%04d'%d.year
month='%02d'%d.month
day='%02d'%d.day
hour='%02d'%d.hour
## All infrared channels used 
for ch in range(8,17):
    channel='C%02d'%ch
    Abi = ABI_Downloader('noaa-goes16',year,month,day,hour,product,channel)
## GLM
Abi = GLM_Downloader('noaa-goes16',year,month,day,hour)
#os.system('cp -R /home/wrf/goes16/* /media/wrf/f0bb40a6-0762-4b60-836c-a7208dcb0cd5/goes16/ ')
#os.system('rm -R /home/wrf/goes16/*')
