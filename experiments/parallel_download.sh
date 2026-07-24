#!/usr/bin/env bash
# Parallel wget for PTB-XL records100/ subdirectories.
# Splits 22 subdirs into 4 parallel workers (~5-6 subdirs each).
set -e
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data"
mkdir -p records100 logs

# Define 4 worker batches (each gets ~5-6 subdirs)
BATCH_1="00000 01000 02000 03000 04000 05000"
BATCH_2="06000 07000 08000 09000 10000 11000"
BATCH_3="12000 13000 14000 15000 16000"
BATCH_4="17000 18000 19000 20000 21000"

download_batch() {
    local batch_id="$1"
    shift
    local subdirs="$@"
    for sd in $subdirs; do
        echo "[batch $batch_id] downloading records100/$sd/..."
        mkdir -p records100/$sd
        cd records100/$sd
        wget -q -N -c -np -nH --cut-dirs=5 \
             -r -A "*.hea,*.dat" \
             "https://physionet.org/files/ptb-xl/1.0.3/records100/$sd/" \
             2>>../../logs/wget_batch_${batch_id}.log
        local n=$(ls *.hea 2>/dev/null | wc -l)
        echo "[batch $batch_id] $sd done: $n .hea files"
        cd ../..
    done
}

# Launch 4 parallel workers
download_batch 1 $BATCH_1 > logs/batch_1.log 2>&1 &
download_batch 2 $BATCH_2 > logs/batch_2.log 2>&1 &
download_batch 3 $BATCH_3 > logs/batch_3.log 2>&1 &
download_batch 4 $BATCH_4 > logs/batch_4.log 2>&1 &

echo "Launched 4 parallel wget workers (PIDs: $(jobs -p | tr '\n' ' '))"
echo "Logs: data/logs/batch_{1,2,3,4}.log"
echo ""
echo "Watching progress (Ctrl-C to detach; wgets keep running):"
while pgrep -f "wget.*physionet" > /dev/null; do
    n=$(find records100/ -name "*.hea" 2>/dev/null | wc -l)
    sz=$(du -sh records100/ 2>/dev/null | awk '{print $1}')
    echo "[$(date +%H:%M:%S)] .hea: $n / 21799 ($((n*100/21799))%) | size: $sz | active wgets: $(pgrep -f 'wget.*physionet' | wc -l)"
    sleep 30
done
echo "All wget workers finished."
