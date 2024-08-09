[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=2.5e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=12000
[ -z "${total_steps}" ] && total_steps=120000
[ -z "${layers}" ] && layers=6
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=24
[ -z "${batch_size}" ] && batch_size=16
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=5
[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${data_path}" ] && data_path='USPTO50K_brief_20230227'
[ -z "${save_path}" ] && save_path='./logs/'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${noise_scale}" ] && noise_scale=0.0
[ -z "${mode_prob}" ] && mode_prob="0,1.0,0.0"
[ -z "${N_vnode}" ] && N_vnode=1
[ -z "${laplacian_pe_dim}" ] && laplacian_pe_dim=0
[ -z "${idx_type}" ] && idx_type=0
[ -z "${use_sep2}" ] && use_sep2="false"
[ -z "${not_sumto2}" ] && not_sumto2="false"
[ -z "${use_class}" ] && use_class="false"
[ -z "${want_h_degree}" ] && want_h_degree="true"
[ -z "${decoder_attn_from_loader}" ] && decoder_attn_from_loader="true"
[ -z "${shufflegraph}" ] && shufflegraph="randomsmiles"
[ -z "${want_decoder_attn}" ] && want_decoder_attn="true"
[ -z "${bpe_tokenizer_path}" ] && bpe_tokenizer_path="none"
[ -z "${charge_h_last}" ] && charge_h_last="false"
[ -z "${flag_old}" ] && flag_old="false"
[ -z "${power}" ] && power=1.0
[ -z "${use_class_encoder}" ] && use_class_encoder="false"
[ -z "${decoder_type}" ] && decoder_type="new"
[ -z "${reduced_head_dim}" ] && reduced_head_dim=8
[ -z "${q_reduced_before}" ] && q_reduced_before="false"
[ -z "${want_emb_k_dynamic_proj}" ] && want_emb_k_dynamic_proj="false"
[ -z "${want_emb_k_dynamic_dropout}" ] && want_emb_k_dynamic_dropout="true"

[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

[ -z "${dict_name}" ] && dict_name='dict_20230310.txt'
[ -z "${position_type}" ] && position_type='sinusoidal'
[ -z "${max_seq_len}" ] && max_seq_len=512

time=$(date "+%Y%m%d-%H%M%S")
echo "time" $time
task_name='NAG2G_unimolplus_uspto_50k'

if [ "$bpe_tokenizer_path" = "none" ]
then
  bpe_tokenizer_path_args=""
else
  bpe_tokenizer_path_args="bpe_"
fi
echo "bpe_tokenizer_path_args: ${bpe_tokenizer_path_args}"

global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`

base_name=${task_name}_b${batch_size}_${n_gpu}_l${layers}_vnode${N_vnode}_wda_${want_decoder_attn}_lp_${laplacian_pe_dim}_${idx_type}_nsum2_${not_sumto2}_sep2_${use_sep2}_cls_${use_class}_hdegree_${want_h_degree}_fo${flag_old}_sg_${shufflegraph}_lr_${lr}_wd_${weight_decay}_mp_${mask_prob}_${bpe_tokenizer_path_args}${time}
save_dir="outputs/${base_name}"
log_save_dir=${save_dir}"/log.log"
mkdir -p ${save_dir}

[ -z "${config_file}" ] && config_file=${save_dir}"/config.ini"

echo $(pwd)/$0
cat $(pwd)/$0 > ${save_dir}/save_orders

export PYTHONPATH=$PYTHONPATH:${PWD}/customized
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $data_path
echo "save_dir" $save_dir
echo "warmup_step" $warmup_step
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "data_folder:"


echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/ //g')
echo ${gpu}
if echo "${gpu}" | grep -q "V100"; then
    gpu_type="--fp16"
    [ -z "${num_workers}" ] && num_workers=5
elif echo "${gpu}" | grep -q "A100"; then
    gpu_type="--bf16"
    [ -z "${num_workers}" ] && num_workers=12
else
    echo "Unsupported GPU: $gpu"
    gpu_type=""
    [ -z "${num_workers}" ] && num_workers=12
fi
echo ${gpu_type}
echo ${num_workers}
if ( $sandwich_ln == "true")
then
  action_args="--sandwich-ln "
else
  action_args=""
fi
echo "action_args: ${action_args}"

if ( $use_sep2 == "true")
then
  use_sep2_args="--use_sep2"
else
  use_sep2_args=""
fi
echo "use_sep2_args: ${use_sep2_args}"

if ( $not_sumto2 == "true")
then
  not_sumto2_args="--not_sumto2"
else
  not_sumto2_args=""
fi
echo "not_sumto2_args: ${not_sumto2_args}"

if ( $use_class == "true")
then
  use_class_args="--use_class"
else
  use_class_args=""
fi
echo "use_class_args: ${use_class_args}"

if ( $want_h_degree == "true")
then
  want_h_degree_args="--want_h_degree"
else
  want_h_degree_args=""
fi
echo "want_h_degree_args: ${want_h_degree_args}"

if ( $decoder_attn_from_loader == "true")
then
  decoder_attn_from_loader_args="--decoder_attn_from_loader"
else
  decoder_attn_from_loader_args=""
fi
echo "decoder_attn_from_loader_args: ${decoder_attn_from_loader_args}"
if ( $want_decoder_attn == "true")
then
  want_decoder_attn_args="--want_decoder_attn"
else
  want_decoder_attn_args=""
fi
echo "want_decoder_attn_args: ${want_decoder_attn_args}"

if ( $charge_h_last == "true")
then
  charge_h_last_args="--charge_h_last"
else
  charge_h_last_args=""
fi
echo "charge_h_last_args: ${charge_h_last_args}"

if ( $flag_old == "true")
then
  flag_old_args="--flag_old"
else
  flag_old_args=""
fi
echo "flag_old_args: ${flag_old_args}"

if ( $use_class_encoder == "true")
then
  use_class_encoder_args="--use_class_encoder"
else
  use_class_encoder_args=""
fi
echo "use_class_encoder_args: ${use_class_encoder_args}"

if ( $q_reduced_before == "true")
then
  q_reduced_before_args="--q_reduced_before"
else
  q_reduced_before_args=""
fi
echo "q_reduced_before_args: ${q_reduced_before_args}"

if ( $want_emb_k_dynamic_proj == "true")
then
  want_emb_k_dynamic_proj_args="--want_emb_k_dynamic_proj"
else
  want_emb_k_dynamic_proj_args=""
fi
echo "want_emb_k_dynamic_proj_args: ${want_emb_k_dynamic_proj_args}"

if ( $want_emb_k_dynamic_dropout == "true")
then
  want_emb_k_dynamic_dropout_args="--want_emb_k_dynamic_dropout"
else
  want_emb_k_dynamic_dropout_args=""
fi
echo "want_emb_k_dynamic_dropout_args: ${want_emb_k_dynamic_dropout_args}"
echo "========================================================================================"


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
torchrun \
      --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
      $(which unicore-train) $data_path --user-dir ./NAG2G \
      --train-subset train --valid-subset valid,test \
      --num-workers ${num_workers} --ddp-backend=no_c10d \
      --task G2G_unimolv2 --loss G2G --arch NAG2G_G2G --encoder-type unimolv2 --noise-scale ${noise_scale} --dict-name ${dict_name} \
      ${gpu_type} --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval 100 --log-format simple \
      --save-interval-updates 5000 --validate-interval-updates 5000 --keep-interval-updates 200 --no-epoch-checkpoints  \
      --save-dir $save_dir --seed $seed \
      --batch-size $batch_size \
      --data-buffer-size ${batch_size} --fixed-validation-seed 11 --batch-size-valid ${batch_size} --required-batch-size-multiple 1 \
	    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 $action_args --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power ${power} \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --encoder-layers $layers --encoder-attention-heads $num_head $add_3d_args $no_2d_args --num-3d-bias-kernel $num_3d_bias_kernel \
      --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
      --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
      ${decoder_attn_from_loader_args} --shufflegraph ${shufflegraph} \
      ${use_sep2_args} ${not_sumto2_args} ${use_class_args} ${want_h_degree_args} ${want_decoder_attn_args} ${charge_h_last_args} ${flag_old_args} ${use_class_encoder_args} \
      --laplacian_pe_dim ${laplacian_pe_dim} --idx_type ${idx_type} \
      --decoder_type ${decoder_type} --reduced_head_dim ${reduced_head_dim} ${q_reduced_before_args} \
      ${want_emb_k_dynamic_proj_args} ${want_emb_k_dynamic_dropout_args} \
      --use_reorder --want_charge_h --add_len 0 --rel_pos --N_vnode ${N_vnode} --max-seq-len ${max_seq_len} \
      --decoder-layers $layers --decoder-embed-dim $hidden_size --decoder-attention-heads $num_head --decoder-ffn-embed-dim $ffn_size \
      --position-type $position_type --config_file ${config_file} \
      --bpe_tokenizer_path ${bpe_tokenizer_path} \
      --auto-regressive --use-decoder --find-unused-parameters \
      | tee -a ${log_save_dir}


echo $(readlink -f ${save_dir})/checkpoint_last.pt
sh valid.sh $(readlink -f ${save_dir})/checkpoint_last.pt
