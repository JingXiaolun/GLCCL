cd ../../

################################################################################# Ablation training ###################################################################################################
################################################################################# Ablation training ###################################################################################################

################################################################ Text_Guided_Flag #####################################################################################################################
## softmax
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Text_Guided_Flag'
#ABLATION_NAME='Softmax'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30400' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--text_guided_flag --aggregation_weights_type softmax \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

# softmax-linear-softmax
JOB_NAME='tc_clip_msrvtt_vit32'
ABLATION_TYPE='Text_Guided_Flag'
ABLATION_NAME='Softmax_Linear_Softmax'
FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30401' \
main_xclip.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--text_guided_flag --aggregation_weights_type softmax_linear_softmax \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32

## mlp-softmax
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Text_Guided_Flag'
#ABLATION_NAME='MLP_Softmax'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30402' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--text_guided_flag --aggregation_weights_type mlp_softmax \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32
################################################################ Text_Guided_Flag #####################################################################################################################




################################################################ Var_Loss_Weight #####################################################################################################################
## Weight (0.1)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.1'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30301' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.1 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.2)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.2'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30302' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.2 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.3)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.3'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30303' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.3 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.4)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.4'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30304' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.4 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.5)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.5'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30305' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.5 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.6)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.6'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30306' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.6 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.7)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.7'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30307' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.7 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.8)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.8'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30308' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.8 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32

## Weight (0.9)
#JOB_NAME='tc_clip_msrvtt_vit32'
#ABLATION_TYPE='Var_Loss'
#ABLATION_NAME='Weight_0.9'
#FILE_DATA_PATH='../../../DataSet/MSRVTT/data/file/'
#VIDEO_DATA_PATH='../../../DataSet/MSRVTT/data/compressed'
#python -m torch.distributed.launch --nproc_per_node=4 --master_port='30309' \
#main_xclip.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
#--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
#--val_csv ${FILE_DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#--data_path ${FILE_DATA_PATH}/MSRVTT_data.json \
#--features_path ${VIDEO_DATA_PATH}/clip4clip_video_frame_input \
#--output_dir ../Model/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--log_dir ../Log/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--visualize_dir ../Visualize/Ablation/${JOB_NAME}/${ABLATION_TYPE}/${ABLATION_NAME} \
#--lr 1e-4 --max_words 32 --max_frames 12 \
#--datatype msrvtt --expand_msrvtt_sentences \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--var_loss_flag --var_loss_weight 0.9 \
#--loose_type --linear_patch 2d --sim_header seqTransf \
#--pretrained_clip_name ViT-B/32
################################################################ Var_Loss_Weight #####################################################################################################################
################################################################################# Ablation training ###################################################################################################
################################################################################# Ablation training ###################################################################################################
