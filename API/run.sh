#!/bin/bash
echo "Running the script: "

#tmux a -t flask-app

#x="source venv/bin/activate"
#eval "$x"
#y=$(eval "$x")
#echo "$y" 

source venv/bin/activate

tmux a -t flask-app

authbind --deep gunicorn3 -w 1 --threads 3 --bind 0.0.0.0:80 app:app > log.txt 2>&1 &

echo "In tmux"

tmux detach
