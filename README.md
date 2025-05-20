# DRBO

To reproduce Table 1, please run
```bash
python main.py \
    --prompt_file prompts/eli5_default.json \
    --eval_file data/eli5_eval_bm25_top100_reranked_oracle.json \
    --dataset_name eli5 \
    --delta_reward true \

python main.py \
    --prompt_file prompts/eli5_default.json \
    --eval_file data/eli5_eval_bm25_top100_reranked_oracle.json \
    --dataset_name eli5 \
    --delta_reward false \

python main.py \
    --prompt_file prompts/asqa_default.json \
    --eval_file data/asqa_eval_gtr_top100_reranked_oracle.json \
    --dataset_name asqa \
    --delta_reward true \

python main.py \
    --prompt_file prompts/asqa_default.json \
    --eval_file data/asqa_eval_gtr_top100_reranked_oracle.json \
    --dataset_name asqa \
    --delta_reward false \
```

To reproduce ALaRM baseline, please run 
```bash
cd alce
python main_alarm.py
```
