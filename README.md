# Efficient Test-Time Alignment for LLMs

## Introduction
This repository implements two test-time / decoding-time alignment methods that improve helpfulness and safety without updating the target LLM’s parameters, by reallocating compute during inference to favor high-reward continuations while maintaining fluency. The code provides end-to-end pipelines for efficient aligned decoding, including segmentation / acceptance logic, model-call accounting, and evaluation hooks used in the accompanying papers.

## [Cascade Reward Sampling for Efficient Decoding-Time Alignment](https://openreview.net/pdf?id=UAA2nWUtVl)
CARDS is a segment-level rejection sampling framework for decoding-time alignment that reduces wasted generation and excessive reward-model (RM) calls. It iteratively proposes semantic segments (not full responses), scores each proposed segment with an external RM, and accepts/rejects the segment via a quasi-rejection criterion—then appends accepted segments to the growing prefix. A key component is uncertainty-based segmentation: segment boundaries are detected using next-token predictive uncertainty (entropy), aiming to ensure each segment is semantically complete so that standard item-level RMs remain accurate on partial prefixes.
![image](./reward_sampling.png)

## [Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner](https://aclanthology.org/2025.emnlp-main.578.pdf)
SSS adapts speculative sampling for weak-to-strong test-time alignment by using an aligned draft model to propose tokens and an unaligned target model to verify them—removing the need for an external RM during decoding. Because aligning the draft increases draft–target distributional shift, SSS modifies (i) the acceptance criterion and (ii) the bonus-token (residual) distribution so that the overall sampling process recovers the RLHF-optimal reward-shifted distribution in theory, while retaining the efficiency benefits of draft-then-verify decoding.

## Recommended Environment
```
python==3.10
torch==2.1.2
transformers==4.46.1
```

## Examples
```
from reward_sampling import RewardSampling
sampler = RewardSampling(access_token=None, llm_dir='argsearch/llama-7b-sft-float32', rm_dir='argsearch/llama-7b-rm-float32')

# Text Generation w/ CARDS
sampler.seg_rs_generate(['###Human: How are you doing today? ###Assistant:'], max_new_token=128)

# Text Generation w/ SSS
sampler.sss_generate(['###Human: How are you doing today? ###Assistant:'], max_new_token=128)

# Reward Scoring
sampler.rm_score(['###Human: How are you doing today? ###Assistant: I am doing well today, thank you! How may I be of service?'])
```

## Benchmark Evaluations
```
CUDA_VISIBLE_DEVICES=0 python evaluation/text_generation.py --method cards --save <your_name> --num-test-prompt 300
CUDA_VISIBLE_DEVICES=0 python evaluation/text_generation.py --method sss --save <your_name> --num-test-prompt 300
```

## Citation
```
@inproceedings{li2025cascade,
  title={Cascade Reward Sampling for Efficient Decoding-Time Alignment},
  author={Bolian Li and Yifan Wang and Anamika Lochab and Ananth Grama and Ruqi Zhang},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://openreview.net/forum?id=QBmxLlmRYG}
}
```
```
@inproceedings{li2025reward,
  title={Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner},
  author={Li, Bolian and Wu, Yanran and Luo, Xinyu and Zhang, Ruqi},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={11479--11489},
  year={2025}
}
```

## Acknowledgement
We thank [ARGS](https://github.com/deeplearning-wisc/args) for their awesome codebase.
