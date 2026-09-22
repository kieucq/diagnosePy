#!/bin/bash -l
#SBATCH -A r00043
#SBATCH -J era5_average
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=256G
##SBATCH --cpus-per-gpu=64
conda deactivate
module load python/gpu/3.12.5
set -x
cd /N/u/ckieu/BigRed200/codex/diagnosePy/src/diagnose_era5/
mm=8
outdir="../../outputs/"
indir="../../data/era5/1946_1960"
level=950

start_year=1940
end_year=1960
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="QC"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

start_year=1940
end_year=1950
indir="../../data/era5/1946_1960"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="QC"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

start_year=1950
end_year=1960
indir="../../data/era5/1946_1960"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="QC"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

start_year=1960
end_year=1970
indir="../../data/era5/1960_1970"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="QC"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

start_year=2000
end_year=2020
indir="../../data/era5/1980_2020"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

#var="T"
#python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}
#python compute_era5_specific_humidity.py ../../outputs/era5_${start_year}_${end_year}_0${mm}_R_${level}.npy \
                                         ../../outputs/era5_${start_year}_${end_year}_0${mm}_T_${level}.npy

start_year=2017
end_year=2020
indir="../../data/era5/1980_2020"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="T"
#python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}
#python compute_era5_specific_humidity.py ../../outputs/era5_${start_year}_${end_year}_0${mm}_R_${level}.npy \
                                         ../../outputs/era5_${start_year}_${end_year}_0${mm}_T_${level}.npy

start_year=1980
end_year=1985
indir="../../data/era5/1980_2020"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="T"
#python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}
#python compute_era5_specific_humidity.py ../../outputs/era5_${start_year}_${end_year}_0${mm}_R_${level}.npy \
                                         ../../outputs/era5_${start_year}_${end_year}_0${mm}_T_${level}.npy

