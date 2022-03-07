#!/bin/bash

echo "Installing dependencies"
mkdir ./assets

echo "Installing ACE 0.9.34"
mkdir ./assets/binaries
wget http://sweaglesw.org/linguistics/ace/download/ace-0.9.34-x86-64.tar.gz -P ./assets/binaries
tar -zxvf ./assets/binaries/ace-0.9.34-x86-64.tar.gz -C ./assets/binaries
rm ./assets/binaries/ace-0.9.34-x86-64.tar.gz

echo "Installing ERG 2018"
mkdir ./assets/grammars
wget http://sweaglesw.org/linguistics/ace/download/erg-2018-x86-64-0.9.34.dat.bz2 -P ./assets/grammars
bzip2 -d ./assets/grammars/erg-2018-x86-64-0.9.34.dat.bz2
