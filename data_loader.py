from datasets import load_dataset
from trl import extract_prompt

def parse_plain(row):
    return row

def parse_json(row):
    if 'chosen' in row and 'rejected' in row:
        return {
            'prompt': row['prompt'][0]['content'],
            'chosen': row['chosen'][0]['content'],
            'rejected': row['rejected'][0]['content'],
        }
    else:
        return {
            'prompt': row['prompt'][0]['content'],
            'response': row['response'][0]['content'],
        }

# prompt, response, chosen, rejected
def PRCR_loader(args, tokenizer, head:int=None, max_length:int=1024):

    # load
    dataset_name = args.data_dir.lower()

    # alpaca_eval.json
    if 'alpaca_eval' in dataset_name:
        data = load_dataset('json', data_files=args.data_dir, split='train')
        data = data.rename_column('instruction', 'prompt')
        data = data.rename_column('output', 'response')
        parse_func = parse_plain

    # walledai/AdvBench
    elif 'advbench' in dataset_name:
        data = load_dataset(args.data_dir, split='train')
        data = data.rename_column('target', 'response')
        parse_func = parse_plain

    # saferlhf.jsonl
    elif 'saferlhf' in dataset_name:
        data = load_dataset('json', data_files=args.data_dir, split='train')
        data = data.rename_column('real', 'chosen')
        data = data.rename_column('generated', 'rejected')
        data = data.map(extract_prompt)
        parse_func = parse_json

    # Skywork/Skywork-Reward-Preference-80K-v0.1
    # pharaouk/ultrafeedback-binarized-preferences-cleaned
    elif 'skywork' in dataset_name or 'ultrafeedback' in dataset_name:
        data = load_dataset(args.data_dir, split='train')
        data = data.map(extract_prompt)
        parse_func = parse_json

    # Dahoas/full-hh-rlhf
    elif 'hh-rlhf' in dataset_name:
        data = load_dataset(args.data_dir, split='test')
        parse_func = parse_plain

    elif 'fineweb' in dataset_name:
        data = load_dataset(args.data_dir, split='train').select(range(100000))
        data = data.rename_column('text', 'prompt')
        parse_func = parse_plain

    else:
        print('\nNOTE: By default, the dataset train split is loaded from HuggingFace, and the keys are assumed to be "prompt" and "response".\n')
        data = load_dataset(args.data_dir, split='train')
        parse_func = parse_plain

    # reformat
    data = data.map(parse_func, remove_columns=set(data.column_names) - {'prompt', 'response', 'chosen', 'rejected'})

    # batch
    data = data.batch(batch_size=args.batch_size)

    # tokenize
    def tokenize_func(row):
        out = tokenizer(row['prompt'], padding=True)
        return {
            'input_ids': out.input_ids,
            'attention_mask': out.attention_mask,
        }
    data = data.map(tokenize_func)

    # filter out long prompts
    data = data.filter(lambda row: len(row['input_ids'][0]) <= max_length)

    # sample a subset
    if head is not None:
        data = data.select(range(head // args.batch_size))

    return data

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='EleutherAI/fineweb-edu-dedup-10b')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.add_special_tokens({'pad_token': '</s>'})

    data = PRCR_loader(args, tokenizer, max_length=512)
    print(data)
    print(len(data))
    length = 0

    for row in tqdm(data):
        length += len(row['input_ids'][0])
    
    print(length / len(data))
