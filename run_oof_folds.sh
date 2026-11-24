#!/bin/zsh
# 5-fold out-of-fold beams for the GER dataset: train on 4 folds, decode the 5th, for every fold.
# Usage: ./run_oof_folds.sh <epochs> [extra train_vsr_lora.py arguments...]
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS=$1; shift
OUT=${OUT:-models/usr2_speaker_lora_v2_folds}
PY=.venv-gf/bin/python
mkdir -p $OUT logs
for K in 0 1 2 3 4; do
  if [ ! -f $OUT/fold_$K.json ]; then
    $PY train_vsr_lora.py --device mps --folds 5 --fold $K --epochs $EPOCHS --save-last "$@" \
      --output $OUT/fold_$K > $OUT/fold_$K.log 2>&1
    # The fold's held-back clips, as links, so eval_wer.py decodes exactly those.
    rm -rf $OUT/fold_${K}_clips && mkdir $OUT/fold_${K}_clips
    $PY -c "import json,os,sys; [os.symlink(os.path.abspath('data/train_sentences/'+n), '$OUT/fold_${K}_clips/'+n) for n in json.load(open('$OUT/fold_$K/training.json'))['val_clips']]"
    $PY eval_wer.py --data $OUT/fold_${K}_clips --decode joint --device cpu --adapter $OUT/fold_$K \
      --json $OUT/fold_$K.json | tail -1
  fi
done
$PY -c "
import json
rows = [dict(r, fold=k) for k in range(5) for r in json.load(open('$OUT/fold_%d.json' % k))]
assert len({r['clip'] for r in rows}) == len(rows) == 260, len(rows)
json.dump(sorted(rows, key=lambda r: r['clip']), open('$OUT/oof_beams.json', 'w'), indent=1, ensure_ascii=False)
e = sum(r['word_errors'] for r in rows); n = sum(r['reference_words'] for r in rows)
print(f'Out-of-fold joint WER: {e / n:.2%} ({e}/{n} words, {len(rows)} clips)')
"
