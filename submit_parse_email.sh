#!/bin/bash
#SBATCH -N 1
#SBATCH --time=16:00:00

export PATH=$HOME/miniconda/bin:$PATH
source activate sherlock

cd $HOME/analyzing-corpora

function log() {
    echo "[$(date +%T)] $1"
}
log "Starting run"
DIR=/var/scratch/jborgdor
tar xzf "$HOME/enron_mail_clean.tgz" -C "$DIR" && \
    python scripts/parse_email.py -d "$DIR/enron_mail3.dic" -p 32 "$DIR/enron_mail_clean" "$DIR/enron_mail3.pkl" && \
    mv "$DIR"/enron_mail3.* data/ && \
    log "Done"
