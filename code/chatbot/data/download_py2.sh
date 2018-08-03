#!/usr/bin/env bash

# download punkt first
python download_punkt.py

python create_ubuntu_dataset.py "$@" --output 'train.csv' 'train'
python create_ubuntu_dataset.py "$@" --output 'test.csv' 'test'
python create_ubuntu_dataset.py "$@" --output 'valid.csv' 'valid'
