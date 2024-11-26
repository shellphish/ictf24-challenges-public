#!bin/bash

service slapd start
ldapadd -x -D "cn=admin,dc=example,dc=com" -w megalol -f /tmp/init.ldif

ldapsearch -x -b "dc=example,dc=com" "(objectClass=person)"

# Move to the webapp folder 
cd /webapp

python3 app.py
