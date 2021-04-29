#!/bin/bash
echo "Running the script: "

tmux a -t flask-app

source venv/bin/activate

authbind --deep gunicorn3 -w 1 --threads 3 --bind 0.0.0.0:80 app:app > log.txt 2>&1 &

echo "In tmux"

tmux detach
