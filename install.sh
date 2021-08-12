#!/bin/bash
wget https://nlp.stanford.edu/data/glove.6B.zip
mkdir data/assets
unzip glove.6B.zip -d data/assets
rm glove.6B.zip

