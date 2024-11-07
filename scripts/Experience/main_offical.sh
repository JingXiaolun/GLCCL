cd ../../

################################################################################# Offical training ###################################################################################################
################################################################################# Offical training ###################################################################################################
# MSR-VTT (ViT-B/32)
JOB_NAME='xclip_msrvtt_vit32'
FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30100' \
main_xclip.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
--output_dir ../Model/Offical/${JOB_NAME} \
--log_dir ../Log/Offical/${JOB_NAME} \
--visualize_dir ../Visualize/Offical/${JOB_NAME} \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32

################################################################################# Offical training ###################################################################################################
################################################################################# Offical training ###################################################################################################
