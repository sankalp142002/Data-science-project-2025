#!/bin/sh
# run-missing.sh  – re-run only missing experiments
# -------------------------------------------------
# Looks for   data/processed/*.npz
# Checks whether results/<model>/metrics_<stem>.json exists
# If not, launches the correct training command.

# ------- model → CLI template (case-statement) ----------------------------
run_model () {
  model="$1" ; npz="$2"
  case "$model" in
    lstm)
      python -m src.models.lstm     --npz-glob "$npz" \
             --epochs 60 --batch 128 --hidden 512 --layers 3 ;;
    gru)
      python -m src.models.gru      --npz-glob "$npz" \
             --epochs 60 --batch 128 --hidden 512 --layers 3 ;;
    bilstm)
      python -m src.models.bilstm   --npz-glob "$npz" \
             --epochs 60 --batch 128 --hidden 512 --layers 3 ;;
    cnnlstm)
      python -m src.models.cnn_lstm --npz-glob "$npz" \
             --epochs 60 --batch 128 --hidden 512 --kernel 5 ;;
    tcn)
      python -m src.models.tcn      --npz-glob "$npz" \
             --epochs 60 --batch 128 --hidden 512 --levels 3 ;;
    tft)
      python -m src.models.tft      --npz-glob "$npz" \
             --epochs 60 --batch 128 --d-model 128 --heads 4 --blocks 2 ;;
    informer)
      python -m src.models.informer --npz-glob "$npz" \
             --epochs 60 --batch 128 --d-model 256 --heads 4 --blocks 3 ;;
    sgp4)
      python -m src.models.sgp4     --npz-glob "$npz" ;;
  esac
}

# -------------- main loop --------------------------------------------------
models="lstm gru bilstm cnnlstm tcn tft informer sgp4"

for npz in data/processed/*.npz ; do
  stem=$(basename "$npz" .npz)              # e.g. 37746_tle_..._w30_h1
  for model in $models ; do
    metrics="results/$model/metrics_${stem}.json"
    if [ ! -f "$metrics" ] ; then
      echo "▶ running $model  →  $(basename "$npz")"
      run_model "$model" "$npz"
    fi
  done
done
