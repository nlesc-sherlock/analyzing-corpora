#!/bin/bash
#SBATCH -N 1
#SBATCH --time=24:00:00
# Compute topic models with LDA, using a different number of topics
# Each LDA is computed in parallel, however, LDA computations with
# more topics will go slower.

# Input filter used
# (removed all words occurring in more than 10% of the documents) 
INFILTER=0.1
# Number of topic models to generate (inclusive)
# Each iteration has the following number of topics:
# floor(1/2 x^2 + x + 3)
SEQ=`seq 0 13`

# Specify two variables, one for local data (slow, compressed) and
# one for scratch (fast, uncompressed)
PROJ_DIR=$HOME/analyzing-corpora
TMPDIR=/var/scratch/jborgdor

# Modify number of topics at each point here
for i in ${SEQ[@]}; do
    TOPICS[$i]=$(($i*$i / 2 + $i + 3))
done

export PATH=$HOME/miniconda/bin:$PATH
source activate sherlock

LOUTDIR=$PROJ_DIR/data/enron_out_$INFILTER
OUTDIR=$TMPDIR/enron_out_$INFILTER
LDIC=$PROJ_DIR/data/filtered_${INFILTER}_5_1000000.dic
DIC=$TMPDIR/filtered_${INFILTER}_5_1000000.dic
ZMATRIX=$PROJ_DIR/data/filtered_${INFILTER}_5_1000000.npz.bz2
MATRIX=$TMPDIR/filtered_${INFILTER}_5_1000000.npz
PROG="python scripts/create_lda.py -d $DIC -s $MATRIX $OUTDIR"

function log() {
    echo "[$(date +%T)] $1"
}

# Prepare scratch directory
cd $PROJ_DIR
log "bunzip2 -ck $ZMATRIX > $MATRIX"
bunzip2 -ck $ZMATRIX > $MATRIX
log "cp $LDIC $DIC"
cp $LDIC $DIC
mkdir -p $OUTDIR

# Do the actual runs, store each PID so we can
# poll the status later
log "Starting runs"
for i in ${SEQ[@]}; do
    log "$PROG -t ${TOPICS[$i]} &"
    $PROG -t ${TOPICS[$i]} &
    pids[$i]=$!
done

# Wait for each PID and after it finishes, store the result back
log "Waiting for tasks to finish"
cd $OUTDIR
for i in ${SEQ[@]}; do
    wait ${pids[$i]}
    # compress result back to home dir
    tar -cjf $LOUTDIR/lda_${TOPICS[$i]}.tar.bz2 lda_${TOPICS[$i]}.pkl*
    log "process ${pids[$i]} done"
done

