---
pretty_name: GateMem
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
  - question-answering
tags:
  - "arxiv:2606.18829"
  - agent-memory
  - memory-governance
  - access-control
  - active-forgetting
  - benchmark
  - multi-principal
  - llm-agent
size_categories:
  - "1K<n<10K"
configs:
  - config_name: medical_episodes
    data_files:
      - split: train
        path: gatemem/data/medical/episodes.jsonl
  - config_name: medical_checkpoints
    data_files:
      - split: train
        path: gatemem/data/medical/checkpoints.jsonl
  - config_name: office_episodes
    data_files:
      - split: train
        path: gatemem/data/office/episodes.jsonl
  - config_name: office_checkpoints
    data_files:
      - split: train
        path: gatemem/data/office/checkpoints.jsonl
  - config_name: education_episodes
    data_files:
      - split: train
        path: gatemem/data/education/episodes.jsonl
  - config_name: education_checkpoints
    data_files:
      - split: train
        path: gatemem/data/education/checkpoints.jsonl
  - config_name: household_episodes
    data_files:
      - split: train
        path: gatemem/data/household/episodes.jsonl
  - config_name: household_checkpoints
    data_files:
      - split: train
        path: gatemem/data/household/checkpoints.jsonl
---

# GateMem

GateMem is a benchmark for evaluating memory governance in multi-principal shared-memory agents.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/662df118b0d404635d7a2b46/8vseSLhB3A39RiHJuy6w0.png" width="75%" alt="GateMem overview">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.18829">📄 Paper</a> •
  <a href="https://rzhub.github.io/GateMem/project.html">🌐 Project Page</a> •
  <a href="https://github.com/rzhub/GateMem">💻 Code</a> •
  <a href="https://rzhub.github.io/GateMem/">🏆 Leaderboard</a> •
  <a href="https://huggingface.co/spaces/Ray368/GateMem-Submit">🚀 Submit Results</a>
</p>

GateMem evaluates three dimensions:

* **Utility**: whether an agent can answer legitimate authorized requests.
* **Access Control**: whether an agent avoids disclosing protected information to unauthorized or over-scoped requesters.
* **Active Forgetting**: whether an agent avoids recovering, confirming, or reconstructing information after explicit deletion requests.

## Domains

GateMem contains four synthetic shared-memory domains:

* `medical`
* `office`
* `education`
* `household`

Each domain contains:

* `episodes.jsonl`: long-form multi-party interaction episodes
* `checkpoints.jsonl`: hidden evaluation checkpoints

## Usage

```python
from datasets import load_dataset

episodes = load_dataset("Ray368/GateMem", "medical_episodes", split="train")
checkpoints = load_dataset("Ray368/GateMem", "medical_checkpoints", split="train")

print(episodes[0])
print(checkpoints[0])
```

For full evaluation, please use the official GateMem benchmark toolkit:

https://github.com/rzhub/GateMem

## License

The dataset is released under the Creative Commons Attribution 4.0 International License.

## Citation

```bibtex
@misc{ren2026gatemembenchmarkingmemorygovernance,
      title={GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents}, 
      author={Zhe Ren and Yibo Yang and Yimeng Chen and Zijun Zhao and Benshuo Fu and Zhihao Shu and Bingjie Zhang and Yangyang Xu and Dandan Guo and Shuicheng Yan},
      year={2026},
      eprint={2606.18829},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.18829}, 
}
```
