#!/usr/bin/python
import sys
import socket
import os
from time import gmtime, strftime
import binascii

UDP_IP = "192.168.23.101"
UDP_PORT = 23181

def networkMessages(command):
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.sendto(command, (UDP_IP, UDP_PORT))
  return ""

def sendICAPpartialBitstream(filename, share=0):
  # READ FILE
  ifname = filename
  f = open(ifname, 'rb');
  sourcefile = (binascii.hexlify(f.read())).upper()
  f.close()
  fb = "  "+ifname+"\n";
  fb += "  read "+str(len(sourcefile)/2)+" bytes\n"

  # TRIM HEADER
  fb +=  "  trimming excess header:\n"
  start_sync=sourcefile.find("000000BB")
  sourcefile = sourcefile[start_sync-8:]
  fb +=  "    remaining "+str(len(sourcefile)/2)+" bytes\n"

  # GROUP BYTES IN WORDS
  words = ["FFFFFFFF"]
  fb +=  "  group bytes in words:\n"
  temp=sourcefile
  i=0
  while len(temp)>0:
    word = temp[0:8]
    temp = temp[8:]
    i+=1
    words.append(str(word)) 

  if share == 1:
    words_new = words[1:len(words)/2]
    words = words_new
    words_new=0
  if share == 2:
    words_new = words[len(words)/2:]
    words = words_new
    words_new=0

  temp=0        # free memory
  sourcefile=0  # free memory

  fb += "  # words "+str(i)+"\n"
  fb += "  # words (list) "+str(len(words))+"\n"
  fb += "    the latter should be the former + 1, due to ICAP disable @0x0\n"

  if len(words) > (2**13) :
    fb += "  ERROR: partial bitstream too large ("+str(len(words))+" vs "+str((2**13))+")\n"
    return fb

  fb += "complete words with address and AE flag\n"
  MESSAGE=""
  for i in range(0,len(words)):
    # first is the address, which is 9-bit => 2 bytes => 4 HEX characters
    address = hex(i)[2:]
    while len(address) < 4:
      address = '0' + address

    # flags
    # 0000: padding   &   AE  &  CSb: 0 (ON)   &   RWn: 0 (write)
    if(i==len(words)-1-1):
      flags = "0000"+"01"+"00"
    else:
      if(i == 0):
        flags = "0000"+"00"+"10"
      else:
        flags = "0000"+"00"+"00"

    flags = hex(int(flags,2))[2:]
    while len(flags) < 2:
      flags = '0' + flags

    line = address +","+ flags +","+ words[i]
    MESSAGE += address + flags + words[i]
    if(i%200==199)|(i==len(words)-1):
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      sock.sendto("D"+binascii.unhexlify(MESSAGE), (UDP_IP, UDP_PORT))
      MESSAGE=""

  return fb

def sendICAPprogram_empty():
  MESSAGE = "000002FFFFFFFF000010FFFFFFFF000204FFFFFFFF000300FFFFFFFF"
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.sendto("D"+binascii.unhexlify(MESSAGE), (UDP_IP, UDP_PORT))
  return ""

def sendICAPprogram_readIDCODE():
  MESSAGE= "000002FFFFFFFF000100FFFFFFFF000200000000BB00030011220044000400FFFFFFFF000500AA99556600060020000000000700200000000008002801800100090020000000000A0020000000000B02FFFFFFFF000C03FFFFFFFF000D01FFFFFFFF000E03FFFFFFFF000F02FFFFFFFF001000300080010011000000000D001200200000000013042000000000140220000000"
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.sendto("D"+binascii.unhexlify(MESSAGE), (UDP_IP, UDP_PORT))
  return ""
#0000 02 FFFFFFFF  <-- !!!! disable ICAP at address 0
#0001 00 FFFFFFFF
#0002 00 000000BB
#0003 00 11220044
#0004 00 FFFFFFFF
#0005 00 AA995566
#0006 00 20000000
#0007 00 20000000
#0008 00 28018001
#0009 00 20000000
#000A 00 20000000
#000B 02 FFFFFFFF
#000C 03 FFFFFFFF
#000D 01 FFFFFFFF
#000E 03 FFFFFFFF
#000F 02 FFFFFFFF
#0010 00 30008001
#0011 00 0000000D
#0012 00 20000000
#0013 04 20000000
#0014 02 20000000


def sendICAPprogram_readFrame(far="00000000", writeFrameToFile=0):

  heading = ["02FFFFFFFF","00FFFFFFFF","00000000BB","0011220044","00FFFFFFFF","00AA995566","0020000000","0030008001","000000000B","0020000000","0030008001","0000000007","0020000000","0020000000","0020000000","0020000000","0020000000","0020000000","0030008001","0000000004","0020000000","0030002001"]

# frame_mask & ae & CSb & RWn (& instruction)

  footing_1 = ["0028006000","00480000A2"]
  footing_2 = ["0020000000"] * 32
  footing_3 = ["0300000000"]
  footing_4a = ["0100000000"] * (81+4) # RWn flag
  footing_4b = ["0900000000"] * 81 # frame mask flag and RWn flag
  footing_5 = ["0300000000","0200000000","0020000000","0030008001","0000000005","0020000000","0030008001","0000000007","0020000000","0030008001","000000000D","0420000000","0020000000"]
  footing = footing_1+footing_2+footing_3+footing_4a+footing_4b+footing_5

  words = heading + [str("00"+far)]+ footing
  fb = str(len(words))
  MESSAGE=""

  if writeFrameToFile == 1:
    f = open('workfile', 'w');
    line = "    rx_d <= x\"%s\"; wait for 8 ns;\n" % binascii.hexlify('D')
    f.write(line)

  for i in range(0,len(words)):
    # first is the address, which is 9-bit => 2 bytes => 4 HEX characters
    address = hex(i)[2:]
    while len(address) < 4:
      address = '0' + address
    MESSAGE += address + words[i]

    if (i+1)%200==0:
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      sock.sendto("D"+binascii.unhexlify(MESSAGE), (UDP_IP, UDP_PORT))

      if ( writeFrameToFile == 1 ):
        for j in range(0, len(MESSAGE),2):

          line = "    rx_d <= x\"%s\"; wait for 8 ns;" % MESSAGE[j:j+2]
          f.write(line+"\n")

      MESSAGE=""

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.sendto("D"+binascii.unhexlify(MESSAGE), (UDP_IP, UDP_PORT))

  if ( writeFrameToFile == 1 ):
    for j in range(0, len(MESSAGE),2):

      line = "    rx_d <= x\"%s\"; wait for 8 ns;" % MESSAGE[j:j+2]
      f.write(line+"\n")

  if writeFrameToFile == 1:
    f.close()

  return fb

if __name__ == "__main__":
  ans = True
  log = ""
  function_feedback = ""

  while ans:
    #if ans != 'f': 
    #  os.system('clear')
    os.system('clear')
    print "FPGA attestation - Interactive menu"
    print """
      r: FPGA runs ICAP program
      Q: FPGA sends a dummy network frame
      F: FPGA sends ICAP feedback

      f: FPGA receives file ICAP program data
          1: pBS cfg01
          2: pBS cfg02
          a: pBS cfg02 part 1/2
          b: pBS cfg02 part 2/2
      i: FPGA receives ICAP program 'read IDCODE'
      e: FPGA receives ICAP program 'empty'

      d: development

      q: quit
    """
    #print "log("+str(log.count('\n'))+"): \n"+log
    log = log.split('\n')
    if len(log) > 5:
      log.pop(len(log)-1)
    log = "\n".join(log)
    print "log (last 5): \n"+log
    print "function feedback: \n"+function_feedback

    ans = raw_input(": ")
    logline=""
    if ans == 'r':
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA runs ICAP program"
      function_feedback = networkMessages('r')
    elif ans == 'd':
      FAR = raw_input("  FAR: ")
      function_feedback = sendICAPprogram_readFrame(FAR);
    elif ans == 'Q':
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA sends a dummy network frame"
      function_feedback = networkMessages('Q')
    elif ans == 'F':
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA sends ICAP feedback"
      function_feedback = networkMessages('F')
    elif ans[0] == 'f':
      if ans[1] == '1':
        filename = "/home/projects/attestation/ML605/bitstreams/cfg01_core_inst00_mod01_partial.bin"
        logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives file ICAP program data (pBS cfg01)"
        function_feedback = sendICAPpartialBitstream(filename)
      elif ans[1] == '2':
        filename = "/home/projects/attestation/ML605/bitstreams/cfg02_core_inst00_mod02_partial.bin"
        logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives file ICAP program data (pBS cfg02)"
        function_feedback = sendICAPpartialBitstream(filename)
      elif ans[1] == 'a':
        filename = "/home/projects/attestation/ML605/bitstreams/cfg02_core_inst00_mod02_partial.bin"
        logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives file ICAP program data (pBS cfg02 part 1/2)"
        function_feedback = sendICAPpartialBitstream(filename, 1)
      elif ans[1] == 'b':
        filename = "/home/projects/attestation/ML605/bitstreams/cfg02_core_inst00_mod02_partial.bin"
        logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives file ICAP program data (pBS cfg02 part 2/2)"
        function_feedback = sendICAPpartialBitstream(filename, 2)

    elif ans == "i":
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives ICAP program 'read IDCODE'"
      function_feedback = sendICAPprogram_readIDCODE()
    elif ans == "e":
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"FPGA receives ICAP program 'empty'"
      function_feedback = sendICAPprogram_empty()

    elif ans == "q":
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"byebye"
      ans=False
    elif ans != "":
      logline += strftime("%Y-%m-%d %H:%M:%S", gmtime())+ ": "+"invalid option ("+ans+")"
    log = "  "+logline+"\n"+log
