# facetap
A Face Detection controlled Tap

## Auto run
Add below to `/etc/crontab`
```
@reboot ubuntu bash -c "cd /home/ubuntu/facetap/flask/ && FLASK_APP=/home/ubuntu/facetap/flask/main.py python2 -m flask run --host=0.0.0.0 > /tmp/t.txt 2>&1"
```
