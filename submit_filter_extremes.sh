#!/bin/bash
#SBATCH -N 1
#SBATCH --time=00:15:00
# From the enron_mail3.pkl dataset, remove words
# occurring in more than A percent of the documents.
export PATH=$HOME/miniconda/bin:$PATH
source activate sherlock

cd $HOME/analyzing-corpora

function log() {
    echo "[$(date +%T)] $1"
}
log "Starting run"
above=( 0.1 0.2 0.3 0.4 0.5 0.6 )
for a in "${above[@]}"; do
    python scripts/filter_extremes.py -a $a -p data/enron_mail3.pkl -d data/enron_mail3.dic data/ &
done

wait
