## [[ICML 2025] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning ](https://arxiv.org/abs/2410.05583)

> [Hyoseo Kim<sup>1,2*](https://sites.google.com/view/hyoseokim), [Dongyoon Han<sup>1</sup>&dagger;](https://dongyoonhan.github.io/), [Junsuk Choe<sup>2</sup>&dagger;](https://sites.google.com/site/junsukchoe/) <br>
> <sup> * Work done during an internship at NAVER AI Lab, &dagger; corresponding authors </sup> <br>
> <sup>1</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab), <sup>2</sup>[Sogang University](https://www.sogang.ac.kr/ko/home)

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2410.05583) [![Paper](https://img.shields.io/badge/Paper-ICML_2025-green)](https://openreview.net/forum?id=ZbWXovStjD)


### Abstract
>Machine unlearning aims to selectively remove specific knowledge from a model. Current methods, such as task arithmetic, rely on fine-tuning models on the forget set, generating a task vector, and subtracting it from the original model. However, we argue the effectiveness of this approach is highly sensitive to hyperparameter selection, necessitating careful validation to identify the best model among many fine-tuned candidates. In this paper, we propose a novel method that leverages all given fine-tuned models rather than selecting a single one. By constructing task vectors from models trained with varied hyperparameters and merging only the components of the task vectors with consistent signs, we perform unlearning by negating the merged task vector from the original model. Given that existing methods also utilize multiple fine-tuned models, our approach delivers more effective unlearning without incurring additional computational costs. We demonstrate the effectiveness of our method on both vision-language models and standard image classification models, showing improved unlearning performance with minimal degradation on the retain set, outperforming state-of-the-art techniques.


### Our Motivation: Hyperparameter Sensitivity in Fine-Tuned Models for Unlearning
- (a) shows performance comparisons with competing methods (+ fine-tuned models alone) in task negation;
- (b, c) illustrates detailed parameter sensitivity across various datasets (b), (c):

  <img width="786" alt="image" src="https://github.com/user-attachments/assets/5a1dc63c-214b-4d8f-8f52-0f5758ad35f7">


### Quick Start 
- Training/evaluation code for CLIP unlearning can be found in [./CLIP_MU/.](https://github.com/naver-ai/negmerge/tree/main/CLIP_MU)
- For a quicker introduction, please refer to the following Jupyter Notebook: [NegMerge.ipynb](https://github.com/naver-ai/negmerge/blob/main/CLIP_MU/src/NegMerge.ipynb)
  
  
### Updates
* (2025/11/24): Code has been released (for CLIP unlearning)
* (2025/05/01): Our paper has been accepted at [ICML 2025](https://icml.cc/)🎉🎉🎉🎉
* (2024/10/09): Our paper has been accepted at [NeurIPS 2024 Workshop on Adaptive Foundation Models](https://adaptive-foundation-models.org/)🎉🎉🎉🎉
* (2024/10/03): Code is under internal review.
* (2024/10/03): [Preprint](https://arxiv.org/abs/2410.05583) has been uploaded.

### Cite
```
@inproceedings{kim2025negmerge,
  title={NegMerge: Sign-Consensual Weight Merging for Machine Unlearning},
  author={Kim, Hyo Seo and Han, Dongyoon and Choe, Junsuk},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

### License
```
NegMerge
Copyright (c) 2025-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
