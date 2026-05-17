# 【CAC'2025 :fire: 】 Text-Video Retrieval With Global-Local Contrastive Consistency Learning

[![CAC](https://img.shields.io/badge/CAC-2025-yellow.svg)](https://doi.org/10.1109/CAC67268.2025.11487249)
This is the implementation of CAC 2025 paper [Text-Video Retrieval With Global-Local Contrastive Consistency Learning](https://doi.org/10.1109/CAC67268.2025.11487249).


## :pushpin: Citation
If you find our method useful in your work, please cite:
```bibtex
@INPROCEEDINGS{11487249,
  author={Jing, Xiaolun and Yang, Xinxing and Yang, Genke},
  booktitle={2025 China Automation Congress (CAC)}, 
  title={Text-Video Retrieval With Global-Local Contrastive Consistency Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1621-1626},
  keywords={Earth Observing System;Artificial satellites;Motion pictures;Broadcasting;Text to video;Videos;Video equipment;Protocols;HTTP;Location awareness;Global-Local Interaction;Contrastive Consistency Learning;Text-Video Retrieval},
  doi={10.1109/CAC67268.2025.11487249}}
```

## :star: Overview
Text-video retrieval aims to find the most semantically similar videos with given text queries. However, since videos contain more diverse content than texts, the main semantics expressed by each text-video pair is often partially relevant. The primary methods involve the utilization of language-video attention module to better align texts and videos. Though effective, this paradigm inevitably introduces prohibitive computational overhead, resulting in inefficient retrieval. In this paper, we propose a simple yet effective method called Global-Local Contrastive Consistency Learning (GLCCL) to achieve texts and videos semantics alignment. Specifically, we design a parameter-free Global-Local Interaction Module (GLIM) to generate semantic-related frame and video features in a text-guided manner. Furthermore, we devise an auxiliary Contrastive Score Consistency (CSC) loss to promote consistency learning among different scores on positive pairs and suppress consistency learning on negative pairs. Extensive experiments on three benchmark datasets demonstrate the superiority and effectiveness of our approach, including MSR-VTT, DiDeMo and VATEX.

![image](https://github.com/JingXiaolun/GLCCL/blob/master/image/motivation.jpg?raw=true)
Figure 1. Illustration of the partially related semantic correspondence between caption (words) and frames from MSR-VTT. Both textual features purely capture sub-regions of frames.

## :herb: Method
![image](https://github.com/JingXiaolun/GLCCL/blob/master/image/framework.jpg?raw=true)
Figure 2. Overview of our proposed Global-Local Contrastive Consistency Learning model (GLCCL). There are two key designs in GLCCL: (1) The global-local interaction module for generating semantically relevant video features with different granularity in a text-guided manner. (2) The contrastive score consistency loss for promoting positive pairs consistency learning and suppressing negative pairs consistency learning.

## :rocket: Quick Start 

### Setup code environment
```bash
pip install -r requirements.txt
```
### Datasets
We train our model on MSR-VTT, DiDeMo and VATEX datasets respectively. Please refer to this [repo](https://github.com/ArrowLuo/CLIP4Clip) for data preparation.

| Datasets  | Google Cloud    | Baidu Yun | Peking University Yun|
|:------:|:------:|:------:|:------:|
| MSR-VTT  | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_)  | [Download](https://pan.baidu.com/share/init?surl=Gdf6ivybZkpua5z1HsCWRA&pwd=enav) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6A028EE7EF5C48A788118B82D6ABE0C5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=MSRVTT&type=anonymous) |
| DiDeMo  | TODO  | [Download](https://pan.baidu.com/s/1Tsy9nb1hWzeXaZ4xr7qoTg?pwd=c842#list/path=%2F) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA14E48D1333114022B736291D60350FA5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=didemo&type=anonymous) |
| VATEX  | TODO  | TODO | TODO |

## How to Run
Download CLIP (ViT-B/32) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
Then, run


**MSR-VTT**

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30400' \
main_glccl.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/file/clip4clip_video_frame_input \
--output_dir ../Model/glccl_msrvtt_vit32 \
--log_dir ../Log/glccl_msrvtt_vit32 \
--visualize_dir ../Visualize/glccl_msrvtt_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \ 
--text_guided_flag --aggregation_weights_type softmax \
--csc_loss_flag --csc_loss_weight 0.5 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

**DiDeMo**

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port='30401' \
main_glccl.py --do_train --num_thread_reader=8 \
--epochs=20 --batch_size=64 --batch_size_val 32 --n_display=50 \
--data_path ../DataSet/DiDeMo/data/compressed/split_file \
--features_path ../DataSet/DiDeMo/data/compressed/split_video \
--output_dir ../Model/glccl_didemo_vit32 \
--log_dir ../Log/glccl_didemo_vit32 \
--visualize_dir ../Visualize/glccl_didemo_vit32 \
--lr 1e-4 --max_words 64 --max_frames 64 \
--datatype didemo \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--text_guided_flag --aggregation_weights_type softmax \
--csc_loss_flag --csc_loss_weight 0.5 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

**VATEX**

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30402' \
main_glccl.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 128 --n_display=50 \
--data_path ../DataSet/VATEX/data/compressed/split_file \
--features_path ../DataSet/VATEX/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/glccl_vatex_vit32 \
--log_dir ../Log/glccl_vatex_vit32 \
--visualize_dir ../Visualize/glccl_vatex_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype vatex \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--text_guided_flag --aggregation_weights_type softmax \
--csc_loss_flag --csc_loss_weight 0.5 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

## :telescope: Experiments
![image](https://github.com/JingXiaolun/GLCCL/blob/master/image/results.jpg?raw=true)

## :bell: Visualization
### T2V Visualization
![image](https://github.com/JingXiaolun/GLCCL/blob/master/image/t2v_visualization.jpg?raw=true)

### V2T Visualization
![image](https://github.com/JingXiaolun/GLCCL/blob/master/image/v2t_visualization.jpg?raw=true)

## :reminder_ribbon: Acknowledgments

The implementation of GLCCL relies on resources from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip") and [CLIP](https://github.com/openai/CLIP "CLIP"). We thank the original authors for their open-sourcing.
