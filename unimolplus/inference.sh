[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=48
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128
[ -z "${num_block}" ] && num_block=4
[ -z "${pos_step_size}" ] && pos_step_size=0.01
[ -z "${gaussian_std_width}" ] && gaussian_std_width=1.0
[ -z "${gaussian_mean_start}" ] && gaussian_mean_start=0.0
[ -z "${guassian_mean_stop}" ] && guassian_mean_stop=9.0


[ -z "${data_path}" ] && data_path="/mnt/vepfs/projects/unimol/pcqm4m_our_mr3d2-ft/"
[ -z "${results_path}" ] && results_path="./infer_confgen"
[ -z "${weight_path}" ] && weight_path="/mnt/vepfs/users/guolin/pcq_weights_v11/guolin-test-pretrain-7-pair-opm-trimul-half-lr2e-4-bs128-n0.2-lp0.3-nb4-mp0.2-ml0.4-mu0.6-gating-pt-ema0.999/checkpoint_best.pt"

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       ./inference.py --user-dir ./unimol/ $data_path --valid-subset $1 \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size 512 \
       --task unimolv2 --loss unimolv2 --arch unimolv2_base \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple \
       --encoder-layers $layers --encoder-attention-heads $num_head --num-3d-bias-kernel $num_3d_bias_kernel \
       --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
       --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout \
       --gaussian-std-width $gaussian_std_width --gaussian-mean-start $gaussian_mean_start --gaussian-mean-stop $guassian_mean_stop \
       --label-prob 0.0 --num-block $num_block --pos-step-size $pos_step_size --pretrain --attn-gating