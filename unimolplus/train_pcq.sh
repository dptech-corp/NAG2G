[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=2e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=150000
[ -z "${total_steps}" ] && total_steps=1500000
[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=48
[ -z "${batch_size}" ] && batch_size=128
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${plddt_loss_weight}" ] && plddt_loss_weight=0.01
[ -z "${pos_loss_weight}" ] && pos_loss_weight=0.2
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128
[ -z "${valid_sets}" ] && valid_sets="valid"
[ -z "${noise}" ] && noise=0.2
[ -z "${label_prob}" ] && label_prob=0.5
[ -z "${mid_prob}" ] && mid_prob=0.1
[ -z "${mid_lower}" ] && mid_lower=0.4
[ -z "${mid_upper}" ] && mid_upper=0.6
[ -z "${num_block}" ] && num_block=2
[ -z "${pos_step_size}" ] && pos_step_size=0.01
[ -z "${gaussian_std_width}" ] && gaussian_std_width=1.0
[ -z "${gaussian_mean_start}" ] && gaussian_mean_start=0.0
[ -z "${guassian_mean_stop}" ] && guassian_mean_stop=9.0
[ -z "${ema_decay}" ] && ema_decay=0.0
[ -z "${pair_embed_dim}" ] && pair_embed_dim=128


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $1
echo "save_dir" $2
echo "warmup_step" $warmup_step
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "valid_sets" $valid_sets

data_path=$1
save_dir=$2
lr=$3
batch_size=$4

save_dir=$save_dir"-lr"$lr"-bs"$batch_size"-n"$noise"-lp"$label_prob-"nb"$num_block-"mp"$mid_prob-"ml"$mid_lower-"mu"$mid_upper

more_args = ""


if [[ -n "${pretrain}" ]] 
then
      more_args=$more_args" --pretrain --best-checkpoint-metric pretrain_loss_by_plddt"
      valid_sets="valid_our"
      save_dir=$save_dir"-pt"
else
      more_args=$more_args" --best-checkpoint-metric loss_by_plddt"
fi

if [[ -n "${input_model}" ]]
then
      more_args=$more_args" --finetune-from-model ${input_model}"
      save_dir=$save_dir"-ft"
fi

# install bc
# apt update
# apt install bc

if (( $(echo "$ema_decay > 0.0" | bc -l) )); then
more_args=$more_args" --ema-decay $ema_decay --validate-with-ema"
save_dir=$save_dir"-ema"$ema_decay
fi

mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
      $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset $valid_sets \
      --num-workers 8 --ddp-backend=c10d \
      --task unimolv2 --loss unimolv2 --arch unimolv2_base \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval 100 --log-format simple \
      --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 50 --no-epoch-checkpoints  \
      --save-dir $save_dir \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid 256 \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 $action_args --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --encoder-layers $layers --encoder-attention-heads $num_head --num-3d-bias-kernel $num_3d_bias_kernel \
      --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob --pair-embed-dim $pair_embed_dim \
      --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
      --plddt-loss-weight $plddt_loss_weight --pos-loss-weight $pos_loss_weight --pos-step-size $pos_step_size \
      --label-prob $label_prob --noise-scale $noise --num-block $num_block \
      --gaussian-std-width $gaussian_std_width --gaussian-mean-start $gaussian_mean_start --gaussian-mean-stop $guassian_mean_stop \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args 