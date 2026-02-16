# [ICLR 2026] AFTER: Mitigating the Object Hallucination of LVLM via Adaptive Factual-Guided Activation Editing

This repository provides the code for the paper [AFTER: Mitigating the Object Hallucination of LVLM via Adaptive Factual-Guided Activation Editing](https://arxiv.org/abs/2601.01957). 

## Abstract
> Large Vision-Language Models (LVLMs) have achieved substantial progress in cross-modal tasks. However, due to language bias, LVLMs are susceptible to object hallucination, which can be primarily divided into category, attribute, and relation hallucination, significantly impeding the trustworthy AI applications. Editing the internal activations of LVLMs has shown promising effectiveness in mitigating hallucinations with minimal cost. However, previous editing approaches neglect the positive guidance offered by factual textual semantics, thereby struggling to explicitly mitigate language bias. To address these issues, we propose Adaptive Factual-guided Visual-Textual Editing foR hallucination mitigation (AFTER), which comprises Factual-Augmented Activation Steering (FAS) and Query-Adaptive Offset Optimization (QAO), to adaptively guide the original biased activations towards factual semantics. Specifically, FAS is proposed to provide factual and general guidance for activation editing, thereby explicitly modeling the precise visual-textual associations. Subsequently, QAO introduces a query-aware offset estimator to establish query-specific editing from the general steering vector, enhancing the diversity and granularity of editing. Extensive experiments on standard hallucination benchmarks across three widely adopted LVLMs validate the efficacy of the proposed AFTER, notably achieving up to a 16.3% reduction of hallucination over baseline on the AMBER benchmark. Our code will be released for reproducibility.

## Table of Contents

1. [Installation](#installation)
2. [Workflow](#workflow)
3. [How to Cite](#how-to-cite)

## Installation
In the root folder of this repo, run the following commands to set things up.
```
conda env create -n after python=3.10
conda activate after
pip install -r requirements.txt
mkdir features
mkdir probes
```

## Workflow
(1) Download the datasets (including the questions and image files) to the `data` folder. We have provided our factual textual descriptions for POPE and AMBER in the corresponding folders.

(2) Get activations by running `bash scripts/get_activations.sh`. All activations are stored in the `features` folder.

(3) Get estimators by running `bash scripts/train_estimator.sh`. The trained estimators will be stored in the `probes` folder.

(4) Conducting editing and evaluating the results by running `bash scripts/inference_editing_single.sh` to perform single inference-time editing on the corresponding LVLM and benchmark. To try out multiple hyperparameter settings, you can run `bash scripts/inference_editing_all.sh`. Read the code to learn about additional options.


## How to Cite

```

@article{wang2026after,
  title={AFTER: Mitigating the Object Hallucination of LVLM via Adaptive Factual-Guided Activation Editing},
  author={Wang, Tianbo and Ma, Yuqing and Liao, Kewei and Zhang, Zhange and Li, Simin and Guo, Jinyang and Liu, Xianglong},
  journal={arXiv preprint arXiv:2601.01957},
  year={2026}
}


```



