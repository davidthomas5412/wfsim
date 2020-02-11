#!/bin/bash
mkdir M2_study
python reference.py --nstar 10000 --chip_rms_tilt 2e-6 --chip_rms_height 4e-6 --reference_file M2_study/reference.pkl
seq 0 99 | gxargs -P5 -n1 -t -I {} python visit.py --M2_amplitude 0.45 --visit_seed {} --visit_file M2_study/visit_{}.pkl --reference_file M2_study/reference.pkl
