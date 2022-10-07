#!/bin/bash

#[[ -x /dev/ttyACM0 ]] || sudo chmod 777 /dev/ttyACM0
cd /home/dev/facetap/flask_hand/
FLASK_APP=/home/dev/facetap/flask_hand/main.py python3 -m flask run --host=0.0.0.0
