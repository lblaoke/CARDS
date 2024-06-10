for i in {1..3}; do
    python eval_ai.py --file_name="./hh_rlhf_output/llama_7b_args300_${i}.jsonl" --output_file="./eval_result/llama_7b_args300_${i}_scores.jsonl"
done

for i in {1..3}; do
    python eval_ai.py --file_name="./hh_rlhf_output/llama_7b_vanilla300_${i}.jsonl" --output_file="./eval_result/llama_7b_vanilla300_${i}_scores.jsonl"
done

for i in {1..3}; do
    python eval_ai.py --file_name="./hh_rlhf_output/mistral_7b_vanilla300_${i}.jsonl" --output_file="./eval_result/mistral_7b_vanilla300_${i}_scores.jsonl"
done

python eval_ai.py --file_name="./hh_rlhf_output/mistral_7b_dpo300_1.jsonl" --output_file="./eval_result/mistral_7b_dpo300_1_scores.jsonl"
python eval_ai.py --file_name="./hh_rlhf_output/mistral_7b_args300_1.jsonl" --output_file="./eval_result/mistral_7b_args300_1_scores.jsonl"
