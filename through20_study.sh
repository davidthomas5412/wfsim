#!/bin/bash
mkdir through20_study
python reference.py --nstar 10000 --chip_rms_tilt 2e-6 --chip_rms_height 4e-6 --reference_file through20_study/reference.pkl
seq 0 99 | gxargs -P5 -n1 -t -I {} python visit.py --M2_amplitude 0.18 --camera_amplitude 0.18 --M1M3_amplitude 0.18 --visit_seed {} --visit_file through20_study/visit_{}.pkl --reference_file through20_study/reference.pkl
