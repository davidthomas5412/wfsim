#!/bin/bash
mkdir through10_study
python reference.py --nstar 10000 --chip_rms_tilt 2e-6 --chip_rms_height 4e-6 --reference_file through10_study/reference.pkl
seq 0 99 | gxargs -P5 -n1 -t -I {} python visit.py --M2_amplitude 0.22 --camera_amplitude 0.22 --M1M3_through10_amplitude 0.22 --visit_seed {} --visit_file through10_study/visit_{}.pkl --reference_file through10_study/reference.pkl
