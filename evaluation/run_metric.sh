# python metric.py --file_name "./hh_rlhf_output/llama_7b_rs100.jsonl" --output_path "./eval_result/metric/llama_7b_rs100.json"
# python metric.py --file_name "./hh_rlhf_output/llama_7b_args100.jsonl" --output_path "./eval_result/metric/llama_7b_args100.json"
# python metric.py --file_name "./hh_rlhf_output/llama_7b_dpo100.jsonl" --output_path "./eval_result/metric/llama_7b_dpo100.json"
# python metric.py --file_name "./hh_rlhf_output/llama_7b_vanilla100.jsonl" --output_path "./eval_result/metric/llama_7b_vanilla100.json"
# python metric.py --file_name "./hh_rlhf_output/llama_7b_rain100.jsonl" --output_path "./eval_result/metric/llama_7b_rain100.json"
# python metric.py --file_name "./hh_rlhf_output/llama_7b_ppo100.jsonl" --output_path "./eval_result/metric/llama_7b_ppo100.json"

# python metric.py --file_name "./hh_rlhf_output/mistral_7b_rs100.jsonl" --output_path "./eval_result/metric/mistral_7b_rs100.json"
# python metric.py --file_name "./hh_rlhf_output/mistral_7b_args100.jsonl" --output_path "./eval_result/metric/mistral_7b_args100.json"
# python metric.py --file_name "./hh_rlhf_output/mistral_7b_dpo100.jsonl" --output_path "./eval_result/metric/mistral_7b_dpo100.json"
# python metric.py --file_name "./hh_rlhf_output/mistral_7b_vanilla100.jsonl" --output_path "./eval_result/metric/mistral_7b_vanilla100.json"
# python metric.py --file_name "./hh_rlhf_output/mistral_7b_rain100.jsonl" --output_path "./eval_result/metric/mistral_7b_rain100.json"
# python metric.py --file_name "./hh_rlhf_output/mistral_7b_ppo100.jsonl" --output_path "./eval_result/metric/mistral_7b_ppo100.json"


#!/bin/bash
# input_folder="./hh_rlhf_output_more"
# output_folder="./eval_result/metric"

# for input_file in $input_folder/*.jsonl; do
#   filename=$(basename -- "$input_file")
#   filename="${filename%.*}"
#   output_file="$output_folder/${filename}.json"

#   python3 metric.py --file_name "$input_file" --output_path "$output_file"
# done

python metric.py --file_name "./hh_rlhf_output_more/mistral_7b_rain100_2.jsonl" --output_path "./eval_result/metric/mistral_7b_rain100_2.json"
python metric.py --file_name "./hh_rlhf_output_more/mistral_7b_rain100_3.jsonl" --output_path "./eval_result/metric/mistral_7b_rain100_3.json"
python metric.py --file_name "./hh_rlhf_output_more/mistral_7b_rain100_4.jsonl" --output_path "./eval_result/metric/mistral_7b_rain100_4.json"
python metric.py --file_name "./hh_rlhf_output_more/mistral_7b_rain100_5.jsonl" --output_path "./eval_result/metric/mistral_7b_rain100_5.json"