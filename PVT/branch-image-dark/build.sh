#!/bin/sh

gcc -S  -o prog.s ./prog.c
gcc -S  -O2 -o progO2.s ./prog.c
