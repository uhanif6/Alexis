#!/usr/bin/python

import socket
import binascii

UDP_IP = "192.168.23.100"
UDP_PORT = 1615

print "UDP target IP:", UDP_IP
print "UDP target port:", UDP_PORT

sock = socket.socket(socket.AF_INET, # Internet
                      socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
	data, addr = sock.recvfrom(1500) # buffer size is 1024 bytes
	print "received message:"#, binascii.hexlify(data)
	data = binascii.hexlify(data)
	parts = [data[i:i+8] for i in range(0, len(data), 8)]
	print "  message length: "+str(len(parts))+" words"
	for i in range(0,len(parts)):
		if parts[i] != "ebebebeb":
			print str(i) + " -> " + str(parts[i])
		#if (i > 55)&(i < 136):
		#	print str(i) + " -> " + str(parts[i])
