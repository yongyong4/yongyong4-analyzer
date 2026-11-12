#!/usr/bin/env bash
# 로컬(맥)에서 VM 에 예측 서버를 올리고 재시작한다.
# 백엔드 레포 deploy.sh(install.sh: 파이썬·MariaDB·yongyong 사용자)를 먼저 끝낸 뒤 실행:
#   ./deploy/deploy.sh ubuntu@1.2.3.4 [ssh 키 경로]
# 최초 1회는 deploy/.env 와 models/ 도 올린다. 이후 models/ 는 서버가 매일 학습하므로 덮어쓰지 않는다
# .env 를 바꿨으면 UPLOAD_ENV=1 ./deploy/deploy.sh ...
set -euo pipefail

HOST=${1:?사용법: deploy.sh user@host [키파일]}
KEY=${2:-}
SSH_OPTS=()
if [ -n "$KEY" ]; then SSH_OPTS=(-i "$KEY"); fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP=/opt/yongyong-model
UP=/tmp/yongyong-model-upload

PARTS=(app sql requirements.txt deploy/yongyong-predict.service deploy/yongyong-train.cron)
FIRST=false
if ! ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$HOST" "test -f /etc/systemd/system/yongyong-predict.service"; then
  FIRST=true
  [ -f "$ROOT/deploy/.env" ] || { echo "deploy/.env 가 없습니다 (deploy/.env.example 참고)"; exit 1; }
  [ -f "$ROOT/models/active.json" ] || { echo "models/active.json 이 없습니다 (먼저 학습)"; exit 1; }
  PARTS+=(models)
fi

echo "== 업로드"
COPYFILE_DISABLE=1 tar -C "$ROOT" --exclude __pycache__ -czf - "${PARTS[@]}" \
  | ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$HOST" "rm -rf $UP && mkdir -p $UP && tar -xzf - -C $UP"
if $FIRST || [ "${UPLOAD_ENV:-0}" = "1" ]; then
  scp ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$ROOT/deploy/.env" "$HOST:$UP/.env"
fi

echo "== 설치·재시작"
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$HOST" "sudo bash -s" <<EOF
set -euo pipefail
mkdir -p $APP/logs
cp -a $UP/app $UP/sql $UP/requirements.txt $APP/
if [ -d $UP/models ]; then cp -a $UP/models $APP/; fi
if [ -f $UP/.env ]; then cp $UP/.env $APP/.env; fi
cp $UP/deploy/yongyong-predict.service /etc/systemd/system/
install -m 644 $UP/deploy/yongyong-train.cron /etc/cron.d/yongyong-train
rm -rf $UP

# LightGBM 이 쓰는 OpenMP
dpkg -s libgomp1 >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y libgomp1
[ -x $APP/.venv/bin/python ] || python3 -m venv $APP/.venv
$APP/.venv/bin/pip install -q -r $APP/requirements.txt

# 분석 테이블 (IF NOT EXISTS 라 다시 실행해도 안전)
mariadb yongyong < $APP/sql/schema.sql

chown -R yongyong:yongyong $APP
chmod 600 $APP/.env
systemctl daemon-reload
systemctl enable yongyong-predict
systemctl restart yongyong-predict
EOF

echo "== 상태"
sleep 10
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$HOST" "systemctl is-active yongyong-predict; curl -s localhost:8000/health; echo"
