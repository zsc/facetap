# facetap
A Face Detection controlled Tap

## Auto run
Add below to `/etc/crontab`
```
@reboot ubuntu bash -c "cd /home/ubuntu/facetap/flask/ && FLASK_APP=/home/ubuntu/facetap/flask/main.py python2 -m flask run --host=0.0.0.0 > /tmp/t.txt 2>&1"
```

## Jetson Nano
### pin mapping
https://stackoverflow.com/questions/61039191/how-to-setup-gpio-pins-in-gpio-tegra-soc-mode-vs-gpio-bcm-mode-using-jetson-nano

The pin #9 here corresponds to #21 on the Jetson Nano.
```
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(9, GPIO.OUT)
```
