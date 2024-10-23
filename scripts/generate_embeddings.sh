#!/bin/bash

# Export the current directory to the PYTHONPATH in .bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:'"${PWD}"'"' >> ~/.bashrc
source ~/.bashrc

# Make the Python script executable
chmod +x src/main.py

# Run the Python script
python3 src/main.py