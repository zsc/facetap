#!/bin/bash

#[[ -x /dev/ttyACM0 ]] || sudo chmod 777 /dev/ttyACM0
cd /home/ubuntu/facetap/flask/
FLASK_APP=/home/ubuntu/facetap/flask/main.py python2 -m flask run --host=0.0.0.0
