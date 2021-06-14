# Fetching pictograms script

The fetching pictograms script fetches all pictograms in Spanish from ARASSAC.
Having the ARASSAC pictograms in a prerequisite for training One-Shot and YOLO models.

Requirements:
 - python 3.8

Execute the following commands:
1. virtualenv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. python ./ARASAAC.py

At the time of writing this README, the Spanish pictograms in ARASSAC are above 12.000 and the execution of the fetching script is taking around 45 min. Be patient, have a break!
