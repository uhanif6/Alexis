#!/usr/bin/python
import socket
import binascii
import re
import sys

from time import sleep
from interactive import sendICAPprogram_readFrame, networkMessages

C_VERBOSE=1
C_SLEEPDURATION=0.01
C_SLEEPDURATION=0.001

def applyMask(receivedFrame, goldenmask):
  temp = ""
  for i in range(0,len(receivedFrame)):
    if goldenmask[i] == '0':
      temp += receivedFrame[i]
    else:
      a = int(receivedFrame[i],16)
      b = int(goldenmask[i],16)
      c = a & ~b
      temp += str(hex(c)[2:])
      #print receivedFrame[i] + " AND not(" + goldenmask[i]+") = "+str(hex(c)[2:])

  return temp

def sendMask(index):
  ifname = "segmenter/impl01/impl01_mask_"+index+".dat"
  fd = open (ifname, 'r')
  mask = fd.read()
  fd.close()
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.sendto("m"+binascii.unhexlify(mask), ("192.168.23.101", 23181))


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "  usage: attestation.py <foldername>"
    exit(1)
  foldername = sys.argv[1]

  # prepare socket and bind to it
  print "Prepare socket"
  UDP_IP = "192.168.23.100"
  UDP_PORT = 1615
  sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
  sock.bind((UDP_IP, UDP_PORT))
  print " UDP target IP:"+str(UDP_IP)
  print " UDP target port:"+str(UDP_PORT)

  # fetch the TODO list
  ifname = "segmenter/output_"+foldername+".csv"
  print "\nRead frame list: "+ifname
  ifd = open(ifname,"r")

  MACok = 0
  MACnok = 0
  MACalmost = 0

  # determine loop boundaries
  print "\nLoop boundaries"
  lower = raw_input("  from (default = 0): ")
  if len(lower) == 0:
    lower = 0
  else:
    lower = int(lower)
    print "    realligning"
    for i in range(0,lower):
      line = (ifd.readline()).rstrip('\n')

  upper = raw_input("  to (default = 28488): ")
  if len(upper) == 0:
    upper = 28488
  else:
    upper = int(upper)
  # lower = 0
  # upper = 28488

  # loop over frame list
  print "\nLoop over frame list"
  ignoreFARtypeList = ("BRAMCONTENT", "LOGIC_OVERHEAD", "BRAMOVERHEAD")
  for i in range(lower,upper):
    
    line = (ifd.readline()).rstrip('\n')
    fields = line.split(",")
    if C_VERBOSE == 1:
      print "  seqnr: "+fields[0]
      print "  FAR: "+fields[1]
      print "  FARtype: "+fields[-1]

    if fields[-1].upper() not in ignoreFARtypeList:

      #sendMask(fields[0][2:])
      #sleep(.1)
      
      # SEND THE ICAP PROGRAM FOR THE TARGETED FRAME
      sendICAPprogram_readFrame(far=fields[1])
      sleep(C_SLEEPDURATION)

      # SEND THE COMMAND TO RUN THE PROGRAM
      networkMessages('r')
      sleep(C_SLEEPDURATION)
      
      # SEND THE COMMAND TO SEND THE FEEDBACK OF THE PROGRAM
      networkMessages('F')
      data, addr = sock.recvfrom(1500) # buffer size is 1024 bytes
      receivedFrame = (binascii.hexlify(data))[0:648]
      receivedNonce = (binascii.hexlify(data))[648:664]
      receivedPadding = (binascii.hexlify(data))[664:688]
      receivedMAC = (binascii.hexlify(data))[688:720]

      if receivedMAC == fields[4]:
        # MAC IS OK
        MACok += 1
      else:
        # MAC IS NOT OK
        # gather what was received
        dbgfname = "segmenter/"+foldername+"/"+foldername+"_"+fields[0][2:]+".dat"
        dfd = open(dbgfname,"r")
        goldenref = dfd.read()
        dfd.close()

        temp = re.sub("[0-9|a-f|A-F]{8} [0-9|a-f|A-F]{8} ([0-9|a-f|A-F]{8})", r"\1", goldenref)
        temp = re.sub("\n", "-", temp)
        temp = re.sub("--.*", "", temp)
        temp = re.sub("-", "", temp)
        goldenframe = re.sub("\n.*", "", temp)
        dfd = open(dbgfname,"r")
        goldenmask = dfd.read()
        dfd.close()

        temp = re.sub("[0-9|a-f|A-F]{8} ([0-9|a-f|A-F]{8}) [0-9|a-f|A-F]{8}", r"\1\n", goldenmask)
        temp = re.sub("\n", "-", temp)
        temp = re.sub("---.*", "", temp)
        temp = re.sub("--", "", temp)
        goldenmask = re.sub("\n", "", temp)

        # apply the mask to the received frame
        receivedFrame = applyMask(receivedFrame, goldenmask)
        #print "    received frame: \n"+receivedFrame

        if goldenframe == receivedFrame:
          print "  frames are equal, but MAC is incorrect"
          MACalmost += 1
        else:
          #print "    received frame: \n"+receivedFrame
          print "    received nonce: "+receivedNonce
          print "    received padding: "+receivedPadding
          print "    received MAC: 0x"+receivedMAC+"\n"
          print "    golden frame: \n"+goldenframe
          print "    golden mask: \n"+goldenmask
          print "    golden MAC: 0x"+fields[4]
          print


          MACnok += 1
          print "  frames are NOT equal"
          #break

      #print "    ok/Nok/almost: "+str(MACok)+"/"+str(MACnok)+"/"+str(MACalmost)+"\n"
      sleep(C_SLEEPDURATION)


  print "MACok: "+str(MACok)
  print "MACnok: "+str(MACnok)

  ifd.close()
