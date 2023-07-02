#!/usr/bin/env bash

queue='vanilla'
njobs=20
outdir=detection_percentages

mkdir -p $outdir/log

for d in $(seq 500 500 2000)
do
        mkdir -p $outdir/$d
	for m in $(seq -2.5 -0.5 -14.0)
	do
		for a in $(seq 1.4 0.2 3.6)
		do
			logfile=./detection_percentages/log/${d}_${m}_${a}.log
			csub -o $logfile -q $queue -n $njobs \
				python ./calc_detection_percentage.py --config ../des.yaml --distance ${d} --abs_mag ${m} --log_a_half ${a}
		done
	done
done



