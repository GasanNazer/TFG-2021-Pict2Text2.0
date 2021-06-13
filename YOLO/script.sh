#!/bin/bash

echo "Executing script"

path=$(echo $1classes.txt | sed 's/\//\\\//g')

echo $path

sed -i "s/names = .*/names = ${path}/g" $1obj.data

echo "$1obj.data"
