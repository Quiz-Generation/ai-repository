#!/bin/bash

port_number="$1"

if [[ -z "$port_number" || ! $port_number =~ ^[0-9]+$ ]]; then
    echo 'port number is required'
    echo 'Using $run_server <port_number: int>'
    exit 2
fi

host_ip="$2"

if [ -z "$host_ip" ]; then
    host_ip='127.0.0.1'
fi

# 절대경로
package_abs_path=$(dirname $(dirname $(realpath "$0")))
cd "$package_abs_path"

export PYTHONPATH=$PYTHONPATH:`pwd`

. venv/bin/activate


# gunicorn -c ../gunicorn.conf.py --log-config ../gunicorn_log.conf  --bind 0.0.0.0:$port_number --log-config ../gunicorn_log.conf --pid ../gunicorn.pid
gunicorn -c /home/jypark/ai-repository/gunicorn.conf.py --log-config /home/jypark/ai-repository/gunicorn_log.conf --bind 0.0.0.0:$port_number --pid /home/jypark/ai-repository/gunicorn.pid

deactivate

