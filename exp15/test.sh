#!/bin/bash

cd `dirname $0`

image_sub_dir="evaluation"

python test.py --image_sub_dir $image_sub_dir
