# run_all_models.sh  ───────────────────────────────────────────
for model in bilstm cnn_lstm gru informer lstm tcn tft; do
  echo "▶ training $model …"
  python -m src.models.$model \
         --npz-glob "data/processed/*.npz" \
         --epochs 120 \
         --batch 128 &          # adjust batch if needed
done
wait                                # ← block until all learners finish

