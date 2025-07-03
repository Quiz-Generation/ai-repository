#!/bin/bash

package_abs_path=$(dirname $0)
pushd $package_abs_path

PID_FILE="`pwd`/../gunicorn.pid"
echo "PID_FILE: $PID_FILE"

if [ -f "$PID_FILE" ]; then
    kill -HUP $(cat "$PID_FILE")
    echo "Gunicorn graceful reload triggered"
else
    echo "PID file not found. Is Gunicorn running?"
    exit 1
fi

popd