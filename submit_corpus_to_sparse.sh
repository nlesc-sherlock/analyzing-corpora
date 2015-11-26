#!/bin/bash
#SBATCH -N 1
#SBATCH --time=02:00:00
# generate sparse matrixes for different filters
# sparse matrixes are much faster to load than
# the full pickle file

export PATH=$HOME/miniconda/bin:$PATH
source activate sherlock

cd $HOME/analyzing-corpora
MYCMD="python scripts/corpus_to_sparse.py"
CORPUS=data/enron_mail3.pkl
function log() {
    echo "[$(date +%T)] $1"
}
log "Starting run"
above=( 0.1 0.2 0.3 0.4 0.5 0.6 )
for a in "${above[@]}"; do
    DIC=data/filtered_${a}_5_1000000.dic
    NPZ=data/filtered_${a}_5_1000000.npz
    log "$MYCMD -d $DIC $CORPUS $NPZ"
    $MYCMD -d $DIC $CORPUS $NPZ
done
