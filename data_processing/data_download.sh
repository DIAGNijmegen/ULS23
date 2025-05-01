#bin/bash
# This script downloads the datasets from Zenodo using the zenodo-download command.
# Download zenodo-downloader using "pip install zenodo-downloader"
# Make sure to set the ZENODO_API_TOKEN environment variable before running this script.
# Usage: ./data_download.sh
# Set the ZENODO_API_TOKEN environment variable
# 
# Part 1
zenodo-download 10035160 -t $ZENODO_API_TOKEN  -w 8 -d .

# Part 2
zenodo-download 10050960 -t $ZENODO_API_TOKEN  -w 8 -d .

# Part 3
zenodo-download 10054306 -t $ZENODO_API_TOKEN  -w 8 -d .

# Part 4
zenodo-download 10054702 -t $ZENODO_API_TOKEN  -w 8 -d .

# Part 5
zenodo-download 10057471 -t $ZENODO_API_TOKEN  -w 8 -d .

# Part 6
zenodo-download 10056235 -t $ZENODO_API_TOKEN  -w 8 -d .

echo -e "\nDownload completed!"
