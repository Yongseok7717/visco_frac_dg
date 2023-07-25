#!/bin/bash

python run_ex2.py -g 20 -G 1 -k 1 -i 1 -I 6 -j 9 -J 10|& tee -i ex2_h_rate_linear.txt
python run_ex2.py -g 20 -G 1 -k 2 -i 1 -I 6 -j 9 -J 10|& tee -i ex2_h_rate_quadratic.txt
python run_ex2.py -g 20 -G 1 -k 3 -i 6 -I 7 -j 3 -J 8|& tee -i ex2_dt_rate_cube.txt