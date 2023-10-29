#!/bin/bash

# Check if .venv exists, if not, create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check for shape_predictor_68_face_landmarks.dat file
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "Please download shape_predictor_68_face_landmarks.dat from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"
fi
