#!/usr/bin/python

import sys
import binascii
import re

if len(sys.argv) != 2:
  print "usage: BSmasking.py <filename.bit>"
  sys.exit()

# READ INPUT FILES
bitfname = sys.argv[1]
mskfname = re.sub(r"\.bit",".msk",sys.argv[1])

f = open(bitfname, 'rb');
bitstream = (binascii.hexlify(f.read())).upper()
f.close()

f = open(mskfname, 'rb');
mskstream = (binascii.hexlify(f.read())).upper()
f.close()


# TRIM HEADERS 
start_sync=bitstream.find("000000BB")
bitstream = bitstream[start_sync-8:]
mskstream = mskstream[start_sync-8:]

# GROUP BYTES IN WORDS
bitwords = []
mskwords = []
tempbit=bitstream
tempmsk=mskstream

while len(tempbit)>0:
  wordbit = tempbit[0:8]
  wordmsk = tempmsk[0:8]
  bitwords.append(str(wordbit)) 
  mskwords.append(str(wordmsk)) 
  tempbit = tempbit[8:]
  tempmsk = tempmsk[8:]

bitstream=0
mskstream=0
tempbit=0
tempmsk=0
wordbit=0
wordmsk=0

print len(bitwords)
print len(mskwords)

frame_x = ["00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00001000","50000000","00004000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","000003c0","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000"]

frame_x = frame_x[53:] + ["EBEBEBEB"]*4000

for i in range(0,len(bitwords)-1):
	print str(bitwords[i]) + "--" + str(mskwords[i]) + "--" + frame_x[i]

	mask=""

	temp = hex(255-int((mskwords[i][0:2]),16))[2:]
	while(len(temp)<2):
		temp = "0"+temp
	mask += temp.upper()
	temp = hex(255-int((mskwords[i][2:4]),16))[2:]
	while(len(temp)<2):
		temp = "0"+temp
	mask += temp.upper()
	temp = hex(255-int((mskwords[i][4:6]),16))[2:]
	while(len(temp)<2):
		temp = "0"+temp
	mask += temp.upper()
	temp = hex(255-int((mskwords[i][6:]),16))[2:]
	while(len(temp)<2):
		temp = "0"+temp
	mask += temp.upper()


	expected_value = str(hex(int(bitwords[i],16)&int(mask,16)))[2:]
	while(len(expected_value)<8):
		expected_value = "0"+expected_value


	if expected_value != frame_x[i]:
		print str(i)+": "+str(bitwords[i]) + "--" + mask + "--" + frame_x[i]+" //"+expected_value+"\\\\  :'( :'( :'( :'(\n"
	else:
		print str(i)+": "+str(bitwords[i]) + "--" + mask + "--" + frame_x[i]+" //"+expected_value+"\\\\\n"


	if (i==28) | (i==(28+81-1)):
		print "= BITST === MASK ==== DATA ====="



sys.exit()

#print start_sync

for i in range(0,start_sync/2):
	b = bfd.read(1)
	m = mfd.read(1)	
	#print binascii.hexlify(b)



while True:
	b = bfd.read(1)
	m = mfd.read(1)
	if (not m) or (not b):
		break;

	b_2 = bin(int(binascii.hexlify(b),16))[2:]
	while len(b_2) < 8:
		b_2 = '0' + b_2

	m_2 = bin(int(binascii.hexlify(m),16))[2:]
	while len(m_2) < 8:
		m_2 = '0' + m_2

	b_i = int(b_2,2)
	m_i = int(m_2,2)

	x_i = b_i & m_i
	x_2 = bin(x_i)[2:]
	while len(x_2) < 8:
		x_2 = '0' + x_2

	line = str(binascii.hexlify(b))+"--"+str(b_2) + " - " + str(m_2) + " - " + str(x_2)
	if b_i != m_i:
		print line+" - 1"
	else:
		print line+" - 0"

