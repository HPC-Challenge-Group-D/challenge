#!/bin/bash
for ((y=3; y<=6; y++))
do
	for ((x=3; x<=6; x++))
	do
		echo "+++ Compiling with x=$((2**x)) | y=$((2**y)):"
		mpicc -o program main.c solver.c -lm -DNX=$((2**x)) -DNY=$((2**x))
		echo "+++ Running:"
		for ((np=1; np<=2; np++))
		do
			mpirun -np $np program
		done
	done
done
