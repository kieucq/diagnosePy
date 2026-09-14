#!/bin/bash
mm=9
var="W"
start_year=1940
end_year=1960
outdir="../../outputs/"
indir="../../data/era5/1946_1960"
level=950
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="Q"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

start_year=1950
end_year=1970
indir="../../data/era5/1970_1980"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="Q"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}


start_year=2000
end_year=2020
indir="../../data/era5/1980_2020"
var="W"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="Q"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}

var="R"
python average.py ${start_year} ${end_year} ${mm} ${var} ${level} --output-dir ${outdir} --data-dir ${indir}
