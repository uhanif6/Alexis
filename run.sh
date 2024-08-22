#!/bin/bash

TARGETNUMBER=1

if [ $TARGETNUMBER -eq 0 ]
then 
  TARGET="impl01"
  BITSTREAM="/home/jvliegen/projects/attestation/ML605/impl01/impl01.bit" 
  BITMASK="/home/jvliegen/projects/attestation/ML605/impl01/impl01.msk"
fi

if [ $TARGETNUMBER -eq 1 ]
then 
  TARGET="config_1"
  BITSTREAM="/home/jvliegen/projects/attestation/ML605/impl01_PA/project_1/project_1.runs/config_1/config_1.bit"
  BITMASK="/home/jvliegen/projects/attestation/ML605/impl01_PA/project_1/project_1.runs/config_1/config_1.msk"
fi

if [ $TARGETNUMBER -eq 2 ]
then 
  TARGET="config_2"
  BITSTREAM="/home/jvliegen/projects/attestation/ML605/impl01_PA/project_1/project_1.runs/config_2/config_2.bit"
  BITMASK="/home/jvliegen/projects/attestation/ML605/impl01_PA/project_1/project_1.runs/config_2/config_2.msk"
fi

export LD_PRELOAD=/opt/usb-driver/libusb-driver.so
ECT_impact.sh $BITSTREAM ML605b

cd /home/jvliegen/vc/attestation/python
rm -Rf segmenter/*
python MACcalculator.py $BITSTREAM $BITMASK
python attestation.py $TARGET
