#!/usr/bin/env bash
# Auto-pipeline: watch parallel download → smoke test → Block A train+eval → Block B → C → D → E → S7+S8.
# Usage: nohup bash experiments/run_when_ready.sh > logs/run_when_ready.log 2>&1 &
set -e
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

EXPECTED=21799
echo "[$(date)] Watching PTB-XL parallel download. Expected: $EXPECTED .hea files."

# 1) Wait for parallel downloads to finish
while true; do
    n=$(find data/records100/ -name "*.hea" 2>/dev/null | wc -l)
    active=$(pgrep -f "wget.*physionet" | wc -l)
    echo "[$(date +%H:%M:%S)] .hea: $n / $EXPECTED ($((n*100/EXPECTED))%) | active wgets: $active"
    if [ "$n" -ge "$EXPECTED" ]; then
        echo "[$(date)] All $EXPECTED .hea files present."
        break
    fi
    if [ "$active" -eq 0 ] && [ "$n" -lt "$EXPECTED" ]; then
        echo "[$(date)] WARN: no active wgets but only $n / $EXPECTED. Waiting 60s in case workers between subdirs..."
        sleep 60
        # Re-check; if still no progress, restart parallel download
        n2=$(find data/records100/ -name "*.hea" 2>/dev/null | wc -l)
        active2=$(pgrep -f "wget.*physionet" | wc -l)
        if [ "$active2" -eq 0 ] && [ "$n2" -le "$n" ]; then
            echo "[$(date)] Restarting parallel download to fill gaps..."
            for sd in $(ls data/records100/); do
                exp=1000
                cur=$(ls data/records100/$sd/*.hea 2>/dev/null | wc -l)
                if [ "$cur" -lt "$exp" ]; then
                    echo "[$(date)] Resuming download for $sd ($cur/$exp)"
                    (cd data/records100/$sd && nohup wget -q -N -c -np -nH --cut-dirs=5 -r -A "*.hea,*.dat" \
                       "https://physionet.org/files/ptb-xl/1.0.3/records100/$sd/" \
                       >>../../logs/wget_resume.log 2>&1 &)
                fi
            done
            sleep 30
        fi
    fi
    sleep 60
done

echo ""
echo "[$(date)] === 2) Smoke test ==="
bash experiments/00_setup.sh 2>&1 | tee logs/00_setup.log

echo ""
echo "[$(date)] === 3) Block A: train baselines + SCP-Soft ==="
bash experiments/run_block_a.sh 2>&1 | tee logs/block_a_train.log

echo ""
echo "[$(date)] === 4) Block A evaluation (post-hoc baselines + bootstrap) ==="
python3 experiments/02_block_a_eval.py 2>&1 | tee logs/block_a_eval.log

echo ""
echo "[$(date)] === 5) Block B: 5-step component-isolation ladder ==="
python3 experiments/03_block_b_ladder.py 2>&1 | tee logs/block_b.log

echo ""
echo "[$(date)] === 6) Block C: stratified analyses ==="
python3 experiments/04_block_c_stratified.py 2>&1 | tee logs/block_c.log

echo ""
echo "[$(date)] === 7) Block D: Pareto + dominance ==="
python3 experiments/05_block_d_pareto.py 2>&1 | tee logs/block_d.log

echo ""
echo "[$(date)] === 8) Block E: clinical case study ==="
python3 experiments/06_block_e_case_study.py 2>&1 | tee logs/block_e.log

echo ""
echo "[$(date)] === 9) Supplements S7, S8 (analysis only) ==="
python3 experiments/07_supplements.py 2>&1 | tee logs/supplements.log

echo ""
echo "[$(date)] === ALL L.3 BLOCKS COMPLETE ==="
echo ""
echo "Next steps in /medical-pipeline:"
echo "  L.4 result-to-claim:   /result-to-claim"
echo "  L.5 review loop:       /journal-review-loop"
echo "  L.6 paper writing:     /paper-plan -> /paper-figure -> /paper-write -> /paper-compile"
