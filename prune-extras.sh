#!/usr/bin/env bash
for model in informer sgp4; do
  for w in 30 60; do
    for h in 1 3 7 30; do
      stem="37746_tle_20250713_170753_last90d_enctrig_w${w}_h${h}"
      rm -f "results/${model}/metrics_${stem}.json"
      rm -f results/checkpoints/${model}/${stem}-best.* 2>/dev/null
      echo "🗑  removed $model → $stem"
    done
  done
done
echo "✅  extras gone – all model folders now have identical stems."
