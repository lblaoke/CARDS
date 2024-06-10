python eval_wintie.py --file_name_red="./hh_rlhf_output/llama_7b_rs100.jsonl" --file_name_blue="./hh_rlhf_output/llama_7b_args100.jsonl" --output_path="./eval_result/wintie/rs_vs_args.jsonl"
python eval_wintie.py --file_name_red="./hh_rlhf_output/llama_7b_rs100.jsonl" --file_name_blue="./hh_rlhf_output/llama_7b_dpo100.jsonl" --output_path="./eval_result/wintie/rs_vs_dpo.jsonl"
python eval_wintie.py --file_name_red="./hh_rlhf_output/llama_7b_rs100.jsonl" --file_name_blue="./hh_rlhf_output/llama_7b_vanilla100.jsonl" --output_path="./eval_result/wintie/rs_vs_vanilla.jsonl"
