#!/bin/bash
echo "Running the script: "

source venv/bin/activate

authbind --deep gunicorn3 -w 1 --threads 3 --bind 0.0.0.0:80 app:app
