import numpy as np
import xarray as xr
#y1, y2 = 1940, 1960
y1, y2 = 1980, 1985
infile1 = f"/N/u/ckieu/Quartz/codex/diagnosePy/data/era5/1980_2020_qc/qc_950_{y1}_{y2}.nc"
yy = 1940 
mm = 10
dd = 1
hh = 0
ds = xr.open_dataset(infile1)
#print(ds)

qc_mean = 0
valid_years_count = 0
for yyyy in range(y1,y2+1):
    print("working on year: ",yyyy)
    try:
        temp = (ds["clwc"]
            .where((ds.valid_time.dt.year == yyyy) & (ds.valid_time.dt.month == mm),drop=True)
            .mean(dim="valid_time").squeeze("pressure_level"))
        qc_mean += temp.values
        valid_years_count += 1
    except Exception as e:
        print(f"Could not process year {yyyy} due to error: {e}. Skipping...")
        continue        
'''
subset = ds["clwc"].sel(
    valid_time=(ds.valid_time.dt.year >= y1) & 
               (ds.valid_time.dt.year <= y2) & 
               (ds.valid_time.dt.month == mm))
qc_mean = subset.mean(dim="valid_time").squeeze("pressure_level").values
'''
np.save(f"era5_{y1}_{y2}_{mm}_QC_950.npy", qc_mean/valid_years_count)
