#!/bin/bash

port_number="$1"

if [[ -z "$port_number" || ! $port_number =~ ^[0-9]+$ ]]; then
    echo 'port number is required'
    echo 'Using $run_server <port_number: int>'
    exit 2
fi

# 서버 실행
package_abs_path=$(dirname $0)  # 현재 'run' 파일의 절대 경로
pushd $package_abs_path/..  # 상위 디렉토리로 이동
export PYTHONPATH=$PYTHONPATH:`pwd`  # 파이썬 환경변수 등록

# # 경로
# cd /home/ubuntu/src/self-marketing-platform-api

source .venv/bin/activate  # 파이썬 env 세팅
gunicorn -c gunicorn.conf.py --bind 0.0.0.0:$port_number --log-config gunicorn_log.conf src.app.main:app
# deactivate

popd


# #!/bin/bash

# port_number="$1"

# if [[ -z "$port_number" || ! $port_number =~ ^[0-9]+$ ]]; then
#     echo 'port number is required'
#     echo 'Using $run_server <port_number: int>'
#     exit 2
# fi

# # 서버 실행
# package_abs_path=$(dirname $0)  # 현재 'run' 파일의 절대 경로
# pushd $package_abs_path  # 파일이 존재하는 경로로 push
# export PYTHONPATH=$PYTHONPATH:`pwd`  # 파이썬 환경변수 등록

# . venv/bin/activate  # 파이썬 env 세팅
# gunicorn -c gunicorn.conf.py --bind 127.0.0.1:$port_number --log-config gunicorn_log.conf
# deactivate

# popd
