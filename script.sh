#!/bin/sh
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q gpgpu
#PBS -l walltime=00:01:00
module add gpu/cuda
cd $PBS_O_WORKDIR
$PBS_O_WORKDIR/program_name > out
