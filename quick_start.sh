python -m SpecGR.run \
  --config 'configs/quick_start.yaml' \
  --eval_mode test \
  --draft_size 50 \
  --num_beams 30 \
  --threshold -1.4 \
  --max_eval_steps 1.0 \
  --devices '[0,1,2,3]'