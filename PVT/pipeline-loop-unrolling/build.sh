#!/bin/sh

gcc -g -fopt-info -o prog ./prog.c
gcc -g -fopt-info -O2 -o progO2 ./prog.c

