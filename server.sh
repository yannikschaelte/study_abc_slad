#!/bin/bash

set -e

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --out) OUT=$VAL ;;
    --file) FILE=$VAL ;;
    --port) PORT=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $OUT ]; then echo "out unset"; exit 1; fi
if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

# load environment
. start_env.sh

hostname > $OUT/ip.txt
echo "Hostname is $(hostname)"

echo "Starting redis server"
redis-server --protected-mode no --port $PORT &

# give server time to start (TODO check if it is up)
sleep 5

# start ncpu-2 workers
echo "Starting workers"
abc-redis-worker --host=localhost --port=$PORT --runtime=24h \
  --processes=46 --daemon=$DAEMON &

sleep 5

# start script
echo "Starting script $FILE"
python $FILE --host=localhost --port=$PORT

