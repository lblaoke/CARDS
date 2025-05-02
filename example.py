from transformers import AutoTokenizer

t1 = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
t2 = AutoTokenizer.from_pretrained('turboderp/Qwama-0.5B-Instruct')

tokens = t1('I don\'t know why I\'m so upset.').input_ids
print(tokens)

tokens = t2('I don\'t know why I\'m so upset.').input_ids
print(tokens)

print(t1.decode(tokens))
