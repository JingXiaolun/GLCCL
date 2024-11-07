cd ../../

################################################################################# X-CLIP + PISA training ###################################################################################################
################################################################################# X-CLIP + PISA training ###################################################################################################

####################################################### MSRVTT-7K DataSet #######################################################################
## ViT-B/32
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_msrvtt_7k_vit32'
#FILE_DATA_PATH='../../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30101' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.7k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input_7k \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA --pale_size 2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## ViT-B/16
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_msrvtt_7k_vit16'
#FILE_DATA_PATH='../../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='30102' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.7k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input_7k \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/16

####################################################### MSRVTT-9K DataSet #######################################################################
# ViT-B/32
PROJECT_NAME='X-CLIP'
JOB_NAME='pisa_msrvtt_vit32'
FILE_DATA_PATH='../../../../DataSet/MSRVTT/data/file/'
VIDEO_DATA_PATH='../../../../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30103' \
main_xclip.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32

## ViT-B/16
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_msrvtt_vit16'
#FILE_DATA_PATH='../../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='30104' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/16

####################################################### DiDeMo DataSet #######################################################################
## ViT-B/32
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_didemo_vit32'
#FILE_DATA_PATH='../../../../DataSet/DiDeMo/data/compressed/split_file'
#VIDEO_DATA_PATH='../../../../DataSet/DiDeMo/data/compressed/split_video'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='30105' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=20 --batch_size=64 --batch_size_val 32 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 64 --max_frames 64 \
#--datatype didemo \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## ViT-B/16
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_didemo_vit16'
#FILE_DATA_PATH='../../../../DataSet/DiDeMo/data/compressed/split_file'
#VIDEO_DATA_PATH='../../../../DataSet/DiDeMo/data/compressed/split_video'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='30106' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=20 --batch_size=16 --batch_size_val 32 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 64 --max_frames 64 \
#--datatype didemo \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/16

####################################################### VATEX DataSet #######################################################################
## ViT-B/32
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_vatex_vit32'
#FILE_DATA_PATH='../../../../DataSet/VATEX/data/compressed/split_file'
#VIDEO_DATA_PATH='../../../../DataSet/VATEX/data/compressed/clip4clip_video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30107' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype vatex \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## ViT-B/16
#PROJECT_NAME='X-CLIP'
#JOB_NAME='pisa_vatex_vit16'
#FILE_DATA_PATH='../../../../DataSet/VATEX/data/compressed/split_file'
#VIDEO_DATA_PATH='../../../../DataSet/VATEX/data/compressed/clip4clip_video_frame_input'
#python -m torch.distributed.launch --nproc_per_node=8 --master_port='30108' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=64 --batch_size_val 128 --n_display=50 \
#--data_path ${FILE_DATA_PATH} \
#--features_path ${VIDEO_DATA_PATH} \
#--output_dir ../../Model/${PROJECT_NAME}/${JOB_NAME} \
#--log_dir ../../Log/${PROJECT_NAME}/${JOB_NAME} \
#--visualize_dir ../../Visualize/${PROJECT_NAME}/${JOB_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype vatex \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --interaction_type PISA \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/16
################################################################################# X-CLIP + PISA training ###################################################################################################
################################################################################# X-CLIP + PISA training ###################################################################################################
