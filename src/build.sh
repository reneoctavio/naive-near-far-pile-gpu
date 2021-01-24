#!/bin/bash

mkdir include
cd include
git clone https://github.com/cusplibrary/cusplibrary.git
cd cusplibrary
git checkout 5ae92304925e635e764474501248f9a10377f316
cd ..
mv cusplibrary/cusp cusp
rm -rf cusplibrary

cd ..
make