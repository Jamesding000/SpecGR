python -m SpecGR.run \
  --config 'configs/quick_start.yaml' \
  --eval_mode test \
  --draft_size 70 \
  --num_beams 50 \
  --threshold -1.2 \
  --devices '[0,1,2,3]'