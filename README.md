# GLCCL: Global-Local Multi-grained Contrastive Consistent Learning for Text-Video Retrieval

Xiaolun Jing, Genke Yang, Jian Chu

This is the official code implementation of the paper "GLCCL: Global-Local Multi-grained Contrastive Consistent Learning for Text-Video Retrieval", the checkpoint will be released soon.

We are continuously refactoring our code, be patient and wait for the latest updates!

## :star: Overview
Text-video retrieval aims to find the most semantically similar videos with given text queries. However, since videos contain more diverse content than texts, the main semantics expressed by each text-video pair is often partially relevant. The primary methods involve the utilization of language-video attention module to better align texts and videos. Though effective, this paradigm inevitably introduces prohibitive computational overhead, resulting in inefficient retrieval. In this paper, we propose a simple yet effective method called Global-Local Contrastive Consistent Learning (GLCCL) to achieve texts and videos semantics alignment. Specifically, we design a parameter-free Global-Local Interaction Module (GLIM) to generate semantic-related frame and video features in a text-guided manner. Furthermore, we devise a Contrastive Score Consistency (CSC) loss to promote consistent learning among different scores on positive pairs and suppress consistent learning on negative pairs. Extensive experiments on the MSR-VTT, DiDeMo and VATEX datasets demonstrate the superiority of our approach. 

![image](https://raw.githubusercontent.com/JingXiaolun/GLCCL/refs/heads/master/image/motivation.jpg)
Figure 1. Illustration of the partially related semantic correspondence between caption (words) and frames from MSR-VTT. Both textual features purely capture sub-regions of frames.

## News

*   **2022.09.20**: Released code

## Requirement

*   [PyTorch](https://pytorch.org/ "PyTorch") version = 1.7.1

*   Install other libraries via

```bash
pip install -r requirements.txt
```

## How to Run

（1）About data download

Please refer to the guides from [CLIP4Clip: Data Preparing](https://github.com/ArrowLuo/CLIP4Clip#:~:text=Data-,Preparing,-For%20MSRVTT).



（2）About the pretrained CLIP checkpoints

You can find more pretrained models in [here](https://github.com/openai/CLIP/blob/main/clip/clip.py "here").

```bash
# download CLIP（ViT-B/32） weight
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# download CLIP（ViT-B/16） weight
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```



（3）About the running scripts

**MSR-VTT**

```bash
# ViT-B/32
sh scripts/run_xclip_msrvtt_vit32.sh

# ViT-B/16
sh scripts/run_xclip_msrvtt_vit16.sh
```

**MSVD**

```bash
# ViT-B/32
sh scripts/run_xclip_msvd_vit32.sh

# ViT-B/16
sh scripts/run_xclip_msvd_vit16.sh
```

**LSMDC**

```bash
# ViT-B/32
sh scripts/run_xclip_lsmdc_vit32.sh

# ViT-B/16
sh scripts/run_xclip_lsmdc_vit16.sh
```

**DiDeMo**

```bash
# ViT-B/32
sh scripts/run_xclip_didemo_vit32.sh

# ViT-B/16
sh scripts/run_xclip_didemo_vit16.sh
```

**ActivityNet**

```bash
# ViT-B/32
sh scripts/run_xclip_actnet_vit32.sh

# ViT-B/16
sh scripts/run_xclip_actnet_vit16.sh
```

## Citation

If you find our method useful in your work, please cite:

```python
@article{Ma2022XCLIP,
  title={{X-CLIP:}: End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval},
  author={Yiwei Ma and Guohai Xu and Xiaoshuai Sun and Ming Yan and Ji Zhang and Rongrong Ji},
  journal={arXiv preprint arXiv:2207.07285},
  year={2022}
}
```

## Acknowledgments

The implementation of X-CLIP relies on resources from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip") and [CLIP](https://github.com/openai/CLIP "CLIP"). We thank the original authors for their open-sourcing.
