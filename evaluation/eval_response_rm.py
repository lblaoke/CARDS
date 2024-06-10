from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import torch
import numpy as np

# Define the model and tokenizer
reward_name = "argsearch/llama-7b-rm-float32"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_name, 
    cache_dir = None,
    num_labels  = 1,
    torch_dtype = torch.float32,
    device_map  = "auto").to('cuda').eval()

tokenizer = AutoTokenizer.from_pretrained(reward_name, cache_dir= None)

# Function to load data from jsonl file and calculate scores
def evaluate_responses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    scores = []
    for line in lines:
        data = json.loads(line)
        prompt = data['prompt']
        response = data['response']

        # Prepare the input for the model
        inputs = tokenizer(prompt, response, return_tensors='pt').to(reward_model.device)
        
        score = reward_model(**inputs).logits[0].cpu().detach()
        scores.append(score.item())
    
    scores = np.array(scores)
    return scores.mean()

# Main function to run the evaluation
if __name__ == "__main__":
    scores = []
    file_path = './hh_rlhf_output/llama_7b_args300_1.jsonl'
    score = evaluate_responses(file_path)
    print(f"Average Score: {score:.4f}")


