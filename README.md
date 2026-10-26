# DecideGen — Behaviourally Aligned Agents and the Context-Constrained Social Simulator

[![status](https://img.shields.io/badge/status-prepared%20·%20release%20deferred-orange)]()
[![license](https://img.shields.io/badge/license-%5Bto%20be%20set%5D-lightgrey)]()

Reference implementation for **Simulating Social Media as It Unfolds: Faithful Collective Dynamics Emerge from Individual Behavioural Alignment** — *submitted to Science Advances*, 2026.

The code reconstructs **individual** social-media behaviour with the **DecideGen** agent and simulates how those individual decisions accumulate into event-level information propagation with the **Context-Constrained Social Simulator (CCSS)**. It then evaluates both levels against real-world events.

Companion dataset: **[decidegen-dataset](https://github.com/xiangru-yin/decidegen-dataset)**.

---

## ⚠️ Release status (please read first)

This repository is **not yet publicly downloadable**.

- The code has been **prepared**; the public release is being finalised alongside dataset de-identification.
- **During peer review**, private access links are provided to the editors and reviewers **on request** (see [Access](#access)).
- **Upon publication**, the code will be made available at **[https://github.com/xiangru-yin/decidegen-code](https://github.com/xiangru-yin/decidegen-code)**.

---

## What is in here

| Component | Role |
|---|---|
| `data/` | Corpus loading, event/profile/history assembly, temporal filtering. |
| `supervision/` | **Underlying action path inference** (Algorithm 1) — candidate-path construction and deterministic action selection. |
| `agents/decider/` | **DecideGen-Decider** — predicts *action · target · emotion*. |
| `agents/generator/` | **DecideGen-Generator** — produces the text when the action requires output. |
| `simulator/` | **CCSS** — the per-user sequential simulation loop, platform-specific ranking/context construction, and state updates. |
| `evaluation/` | Individual-level and collective-level metrics. |
| `configs/` | Training and simulation configuration files. |

### Pipeline

```
public records ──▶ traceable bilingual corpus ──▶ underlying action path inference
                                                        │  (action · target · content · emotion)
                                                        ▼
                                   supervision sets ──▶ DecideGen (Decider + Generator)
                                                        │
                                                        ▼
                                     CCSS per-user simulation ──▶ propagation network + trajectories
                                                        │
                                                        ▼
                                                  evaluation metrics
```

---

## Models

DecideGen is built on a family of **domain-specific foundation models** developed by continual pre-training on a 24-billion-token media and public-opinion corpus; the lightweight variant follows the same pipeline on one fifth of that corpus (Supplementary Methods):

| Model | Base | Used as |
|---|---|---|
| **Domain-specific foundation model** (full) | Qwen3-8B-Base | **DecideGen-Generator** |
| **Domain-specific foundation model** (lightweight) | Qwen3-1.7B-Base | **DecideGen-Decider** |

- **Decider** receives the user profile, event background and current environmental context, encodes them (with structured behavioural labels) into decision tokens, predicts the label sequence, and decodes it into *action · target · sentiment*. Trained with cross-entropy over the reference decision tokens.
- **Generator** is activated only for content-producing actions and additionally receives the Decider's predicted action · target · emotion. Trained with cross-entropy over the reference content tokens.

The two modules run **sequentially** inside CCSS: the Decider fixes *what / with whom / in what sentiment*, and the Generator then realises the content — so the text is explicitly constrained by the preceding decision.

---

## Requirements

- **Python** ≥ 3.9, **PyTorch** (bf16-capable GPU)
- **Hugging Face Transformers**
- **LLaMA-Factory v0.9.4.dev0** (training entry point; used unmodified, all hyperparameters from config)
- **DeepSpeed** (ZeRO-2)
- *(Optional, for baselines)* **DeepSeek-V4-flash** accessed through the DeepSeek Chat API (`deepseek-chat` endpoint), and the official releases of PolicySim / OASIS / HiSim

### Training configuration

Trained on **8 × NVIDIA A100** GPUs.

| Setting | Decider | Generator |
|---|---|---|
| Learning rate | 5 × 10⁻⁶ | 5 × 10⁻⁶ |
| Epochs | 2.0 | 2.0 |
| LR scheduler | Cosine | Cosine |
| Warm-up ratio | 0.1 | 0.1 |
| Per-device batch size | 8 | 4 |
| Gradient accumulation | 2 | 2 |
| Effective global batch | **128** | **64** |

Common: bf16 mixed precision · DeepSpeed ZeRO-2 · max sequence length 4,096 · optimizer AdamW · checkpoints once per epoch · 1% of the corpus for validation · training seed = framework default (**42**); the simulator fixes its Python and NumPy generators at **1**.

```bash
# install
pip install -r requirements.txt        # [to be published]

# train the Decider
llamafactory-cli train configs/decider.yaml

# train the Generator
llamafactory-cli train configs/generator.yaml
```

---

## Usage

```bash
# 1. build supervision (underlying action path inference)
python -m supervision.build --config configs/supervision.yaml \
    --records <records/> --users <users/> --history <history/>

# 2. run CCSS for one or more events
python -m simulator.run --config configs/simulator.yaml \
    --event <event_id> --output <output_dir/>

# 3. evaluate against the observed event
python -m evaluation.run --config configs/evaluate.yaml \
    --simulated <output_dir/> --observed <records/> --out report.md
```

*(Command names reflect the intended public layout and will be finalised with the release.)*

---

## Evaluation

**Individual level** — does DecideGen reproduce a real user's sequential decisions and content?

- Decision: **ACC** (Accuracy), **Weighted-F1**, **EPM** (Exact Prefix-Match Rate — every prediction in an evaluated prefix of a behaviour chain must match).
- Generation: **BERT-Sim** (BERT-based cosine similarity), **ROUGE-L** (F-measure).

**Collective level** — does CCSS reconstruct propagation structure and dynamics?

- Structure: **KSD** (Kolmogorov–Smirnov distance for degree distribution), **DDKS** (depth-distribution KS distance), **WID** (normalized Wiener-index distance) — *lower is better*.
- Collective sentiment: **PCC_sen**, **RMSE_sen**, **WD** (Wasserstein distance) — *higher PCC_sen, lower RMSE_sen / WD*.
- Public attention: **PCC_att**, **RMSE_att**, **PHRE** (peak-height relative error) — *higher PCC_att, lower RMSE_att / PHRE*.

Evaluation is performed on **15 held-out events** (10 Chinese · 5 English). Full metric definitions and formulas are in the **Supplementary Methods**.

---

## Data

This repository requires the companion dataset, which is **not yet public** (see its repository for the release schedule). The dataset supplies the corpus, profiles, historical posts, follow edges, and the supervision sets. Raw Weibo/Twitter content is **not** redistributed — only derived, de-identified records are released.

---

## Release status

| Artefact | Status |
|---|---|
| Code (this repo) | prepared · **private until publication** · **[https://github.com/xiangru-yin/decidegen-code](https://github.com/xiangru-yin/decidegen-code)** |
| Dataset | prepared · **private until publication** · **[https://github.com/xiangru-yin/decidegen-dataset](https://github.com/xiangru-yin/decidegen-dataset)** |
| Model checkpoints | DecideGen-Decider / DecideGen-Generator weights derived from Qwen3-1.7B-Base / Qwen3-8B-Base (Apache-2.0); release mode **[available on request] |

---

## License

**[Apache-2.0; consistent with the Qwen3 base models.]** Third-party components (LLaMA-Factory, Transformers, DeepSpeed, baselines) remain under their own licences.

---

## Citation

```bibtex
@article{yin2026decidegen,
  title   = {Simulating Social Media as It Unfolds: Faithful Collective Dynamics Emerge from Individual Behavioural Alignment},
  author  = {Yin, Xiangru and Li, Huan and Tong, Shiyu and Chen, Zizhao and Ran, Shuyang and Zheng, Nanning and Wei, Ping},
  journal = {Science Advances},
  year    = {2026},
  note    = {Under review},
  doi     = {[DOI — to be added upon publication]}
```

---

## Contact

**Xiangru Yin** — xiangruyin@stu.xjtu.edu.cn · State Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Institute of Artificial Intelligence and Robotics, Xi'an Jiaotong University, Xi'an 710049, Shaanxi, China

**Corresponding author** — Ping Wei, pingwei@xjtu.edu.cn

*(Remaining items — repository DOIs and the data licence — will be filled in before public release.)*
