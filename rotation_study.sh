#!/bin/bash
mkdir rotation_study
python reference.py --nstar 10000 --chip_rms_tilt 2e-6 --chip_rms_height 4e-6 --reference_seed 57 --reference_file rotation_study/reference.pkl
seq 0 499 | gxargs -P5 -n1 -t -I {} python visit.py --M2_amplitude 0.18 --camera_amplitude 0.18 --M1M3_amplitude 0.18 --visit_seed {} --rot_seed {} --star_seed {} --nstar 10000 --visit_file rotation_study/visit_{}.pkl --reference_file rotation_study/reference.pkl
