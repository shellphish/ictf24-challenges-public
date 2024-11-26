1) parse pcap file in python (can use, for example, scapy)
2) extract features from the packets (make sure dst port is in there)
3) train a classifier (random forest seems to work well. Can use sklearn for this)
4) run model on packets
5) extract content from anomalous packets
6) concat content together and decode using base64. This yields the flag

Depending on the accuracy of the classifier, some additional filtering might be needed.
For example, checking whether the extracted content is valid base64, checking whether there are too many repeated characters, etc.


Alternative solution:
1) inspect packets manually
2) find dst port pattern of anomalous packets and extract flag that way
