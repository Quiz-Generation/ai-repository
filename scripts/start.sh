#!/bin/bash

port_number="$1"

if [[ -z "$port_number" || ! $port_number =~ ^[0-9]+$ ]]; then
    echo 'port number is required'
    echo 'Using $run_server <port_number: int>'
    exit 2
fi

host_ip="$2"

if [ -z "$host_ip" ]; then
    # host_ip='127.0.0.1'
    host_ip='0.0.0.0'
fi

# package_abs_path=$(dirname $0)
# pushd $package_abs_path
# 절대경로
package_abs_path=$(dirname $(dirname $(realpath "$0")))
cd "$package_abs_path"

export PYTHONPATH=$PYTHONPATH:`pwd`

. venv/bin/activate

gunicorn -c gunicorn.conf.py --log-config gunicorn_log.conf --bind $host_ip:$port_number --pid ./gunicorn.pid

deactivate

popd
