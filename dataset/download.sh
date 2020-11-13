#!/bin/bash

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

function download () {
 arrayName=($1)
 arrayID=($2)
 count=${#arrayName[@]}

 for i in `seq 1 $count`
 do
  echo -e "${GREEN} File: ${NC}" ${arrayName[$i-1]}
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${arrayID[$i-1]}" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${arrayID[$i-1]}" -o $3/${arrayName[$i-1]}
 done
}

if [ ! -d "./control" ]; then
 mkdir -p control
fi
if [ ! -d "./localization" ]; then
 mkdir -p localization
fi

controlID="1t83kqIgmuCy9kJbXsoCvTnFETHmznHUs 1GFt2P5vJU1_CiwyadSwxkw-YSgoiDf6d 1ErUMb8Hk2TRykhGBMFQehYz8Cx02NhIW 1YNhewFnK2pu6hVFKhzGYmkV9yiUboq7Q"
controlName="raw_test.csv raw_train.csv test.csv train.csv"
pathControl=./control

echo -e "${BLUE}Downloading Control Dataset...${NC}"
download "$controlName" "$controlID" "$pathControl"

localizationID="16MzCUfqjvQh_c0fvVHWXSAR7yoeKKfeC 1dKjZzoCHofbLpfqgeCCe4chZ9SBAkoGS 1Vmcw1rNtAUzgwvNc6TG4qFi264pEcTgS 14--BpRMFZ_XfRpXYI_Bthr3ZVx8kjZpE"
localizationName="raw_test.csv raw_train.csv test.csv train.csv"
pathLocalization=./localization

echo -e "${BLUE}Downloading Localization Dataset...${NC}"
download "$localizationName" "$localizationID" "$pathLocalization"
