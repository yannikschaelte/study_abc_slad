#!/bin/bash
#SBATCH --account=fitmulticell
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=batch
#SBATCH --time=24:00:00

set -e

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --account) ACCOUNT=$VAL ;;
    --nodes) NODES=$VAL ;;
    --out) OUT=$VAL ;;
    --file) FILE=$VAL ;;
    --port) PORT=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $ACCOUNT ]; then echo "account unset"; exit 1; fi
if [ -z $NODES ]; then echo "nodes unset"; exit 1; fi
if [ -z $OUT ]; then echo "out unset"; exit 1; fi
if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

# load environment
. start_env.sh

# start server
srun \
  --nodes=1 --ntasks=1 \
  --output=$OUT/out_slurm_worker_$IW.txt --error=$OUT/err_slurm_worker_$IW.txt \
  server.sh \
  --out=$OUT \
  --port=$PORT --daemon=$DAEMON --file=$FILE &

# give server time to start
sleep 10

# retrieve server id
HOSTNAME=`cat $OUT/ip.txt | tr '\n' ' '`
HOST=$(host $HOSTNAME | awk '{ print $4 }')
echo "Server running on IP $HOST"

# start workers
WORKERS=$((NODES-2))
echo "Starting $WORKERS workers"
for IW in $(seq $WORKERS); do
  srun \
    --nodes=1 --ntasks=1 \
    --output=$OUT/out_slurm_worker_$IW.txt --error=$OUT/err_slurm_worker_$IW.txt \
    worker.sh \
    --host=$HOST --port=$PORT --daemon=$DAEMON &
done

# start one more worker
srun \
  --nodes=1 --ntasks=1 \
  --output=$OUT/out_slurm_worker_$IW.txt --error=$OUT/err_slurm_worker_$IW.txt \
  worker.sh \
  --host=$HOST --port=$PORT --daemon=$DAEMON
