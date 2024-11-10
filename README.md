# GLCCL: Global-Local Multi-grained Contrastive Consistent Learning for Text-Video Retrieval

[Xiaolun Jing](https://scholar.google.com/citations?hl=zh-CN&user=LsozN5kAAAAJ), Genke Yang, Jian Chu

This is the official code implementation of the paper "GLCCL: Global-Local Multi-grained Contrastive Consistent Learning for Text-Video Retrieval", the checkpoint will be released soon.

We are continuously refactoring our code, be patient and wait for the latest updates!

## :star: Overview
Text-video retrieval aims to find the most semantically similar videos with given text queries. However, since videos contain more diverse content than texts, the main semantics expressed by each text-video pair is often partially relevant. The primary methods involve the utilization of language-video attention module to better align texts and videos. Though effective, this paradigm inevitably introduces prohibitive computational overhead, resulting in inefficient retrieval. In this paper, we propose a simple yet effective method called Global-Local Contrastive Consistent Learning (GLCCL) to achieve texts and videos semantics alignment. Specifically, we design a parameter-free Global-Local Interaction Module (GLIM) to generate semantic-related frame and video features in a text-guided manner. Furthermore, we devise a Contrastive Score Consistency (CSC) loss to promote consistent learning among different scores on positive pairs and suppress consistent learning on negative pairs. Extensive experiments on the MSR-VTT, DiDeMo and VATEX datasets demonstrate the superiority of our approach. 

![image](https://raw.githubusercontent.com/JingXiaolun/GLCCL/refs/heads/master/image/motivation.jpg)
Figure 1. Illustration of the partially related semantic correspondence between caption (words) and frames from MSR-VTT. Both textual features purely capture sub-regions of frames.

## :herb: Method
![image](https://raw.githubusercontent.com/JingXiaolun/GLCCL/refs/heads/master/image/framework.jpg?token=GHSAT0AAAAAACY4XFD745BGL6VDGFQ3A6MQZZNYPXQ)
Figure 2. Overview of our proposed Global-Local Contrastive Consistent Learning model (GLCCL). There are two key designs in GLCCL: (1) The global-local interaction module for generating semantically relevant video features with different granularity in a text-guided manner. (2) The contrastive score consistency loss for promoting positive pairs consistent learning and suppressing negative pairs consistent learning.

## :mag: Usage 

### Requirement
```bash
pip install -r requirements.txt
```
### Datasets
We train our model on MSR-VTT, DiDeMo and VATEX datasets respectively. Please refer to this [repo](https://github.com/ArrowLuo/CLIP4Clip) for data preparation.

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
--var_loss_flag --var_loss_weight 0.5 \
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
--var_loss_flag --var_loss_weight 0.5 \
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
--var_loss_flag --var_loss_weight 0.5 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

## :telescope: Experiments
![image](https://raw.githubusercontent.com/JingXiaolun/GLCCL/refs/heads/master/image/results.jpg?token=GHSAT0AAAAAAC2HXBYXYIEQTCEXLC7AD3G4ZZQJ55Q)

## :wrench: Visualization
![image](https://raw.githubusercontent.com/JingXiaolun/GLCCL/refs/heads/master/image/results.jpg?token=GHSAT0AAAAAAC2HXBYXYIEQTCEXLC7AD3G4ZZQJ55Q)

## Acknowledgments

The implementation of GLCCL relies on resources from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip") and [CLIP](https://github.com/openai/CLIP "CLIP"). We thank the original authors for their open-sourcing.
