#!/bin/bash

# convenience wrapper around sbatch submission script

set -e

ACCOUNT="fitmulticell"
DAEMON="True"
TIME="24:00:00"

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --account) ACCOUNT=$VAL ;;
    --nodes) NODES=$VAL ;;
    --time) TIME=$VAL ;;
    --file) FILE=$VAL ;;
    --port) PORT=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $NODES ]; then echo "nodes unset"; exit 1; fi
if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi

OUT=`mktemp -d -p $PWD -t "out_XXXX"`
echo "Output folder: $OUT"

# run batch script
sbatch \
    --account=$ACCOUNT \
    --nodes=$NODES --ntasks=$NODES \
    --output=$OUT/out_slurm.txt --error=$OUT/err_slurm.txt \
    --time=$TIME \
  sbatch_job.sh \
    --account=$ACCOUNT --nodes=$NODES \
    --out=$OUT --file=$FILE --port=$PORT --daemon=$DAEMON
