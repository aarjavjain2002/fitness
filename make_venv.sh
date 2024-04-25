#!/bin/bash

ENVNAME="fitness-venv"
if [ -d "$ENVNAME" ]; then
    echo "Removing previous environment: $ENVNAME"
    rm -Rf $ENVNAME  # Use 'rm -Rf' for forceful recursive deletion.
else
    echo "No previous environment - ok to proceed"
fi

python3.11 -m venv $ENVNAME
source $ENVNAME/bin/activate
pip install --upgrade pip  # Always use 'pip' after activating the venv
pip install wheel  # Install wheel to ensure smooth installations of other packages

pip install -r requirements.txt  # Install required packages from requirements.txt
python3 --version  # Check the Python version
