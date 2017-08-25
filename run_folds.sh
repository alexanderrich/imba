#!/bin/bash

purge
for i in {1..9}
do
	python analysis_sh1ng.py $i >> sh1ng_output.txt
  purge
done
