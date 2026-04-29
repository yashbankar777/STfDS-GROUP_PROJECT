#!/bin/bash

# Config
OUTPUT_DIR="data/demand"
BASE_URL="https://www.aemo.com.au/aemo/data/nem/priceanddemand"
REGIONS=("NSW1")  # Add VIC1, QLD1, SA1, TAS1 if you need them
START_YEAR=2022
END_YEAR=2025

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit

echo "Downloading AEMO price & demand CSVs to $(pwd)"
echo "Attempting full 12 months for $START_YEAR-$END_YEAR inclusive"

for region in "${REGIONS[@]}"; do
  for year in $(seq $START_YEAR $END_YEAR); do
    for month in {01..12}; do
      FILE="PRICE_AND_DEMAND_${year}${month}_${region}.csv"
      URL="$BASE_URL/$FILE"
      
      echo "Trying $FILE"
      wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
           --referer="https://aemo.com.au/" \
           -nc -q --show-progress \
           "$URL"
      
      # 404s expected for months that don't exist yet
      sleep 1
    done
  done
done

echo ""
echo "Download complete. Files in $OUTPUT_DIR:"
ls PRICE_AND_DEMAND_*.csv 2>/dev/null | wc -l
echo "files downloaded"