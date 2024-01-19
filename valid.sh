[ -z "${MASTER_PORT}" ] && MASTER_PORT=12345
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
# n_gpu=1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
# [ -z "${batch_size}" ] && batch_size=1
batch_size=1
[ -z "${seed}" ] && seed=1
[ -z "${len_penalty}" ] && len_penalty=0.0
[ -z "${beam_size}" ] && beam_size=10
[ -z "${beam_size_second}" ] && beam_size_second=5
[ -z "${beam_head_second}" ] && beam_head_second=2
# SimpleGenerator SequenceGeneratorBeamSearch SequenceGeneratorBeamSearch_test
[ -z "${search_strategies}" ] && search_strategies=SimpleGenerator
[ -z "${temperature}" ] && temperature=1
[ -z "${num_workers}" ] && num_workers=5

path=$1
results_path=$(echo "$path" | sed 's/\.pt$//')
config_file=$(dirname "$results_path")/config.ini
# u50 uf g50 gf
if echo "${results_path}" | grep -q "unimolplus_uspto_50k"; then
    [ -z "${model_infer_type}" ] && model_infer_type=u50
else
    echo "Unsupported infer name: $results_path"
    model_infer_type=$2
fi
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
time=$(date "+%Y%m%d-%H%M%S")
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/ //g')
unimol_version=$(pip show unimol | grep Version | awk '{print $2}')

echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="

echo "time" $time
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $data_path
echo "save_dir" $save_dir
echo "seed" $seed
echo "gpu" $gpu
echo "unimol_version" $unimol_version

if echo "${gpu}" | grep -q "V100"; then
    [ -z "${num_workers}" ] && num_workers=5
elif echo "${gpu}" | grep -q "A100"; then
    [ -z "${num_workers}" ] && num_workers=12
else
    echo "Unsupported GPU: $gpu"
    [ -z "${num_workers}" ] && num_workers=12
fi
echo "num_workers" ${num_workers}
 
case $model_infer_type in
  u50)
    [ -z "${data_path}" ] && data_path='USPTO50K_brief_20230227'
    task=G2G_unimolv2
    encoder_type=unimolv2
    echo "u50"
    ;;
  *)
    echo "未知取值"
    ;;
esac

data_folder=$(basename $data_path)
echo $data_folder
infer_save_name=smi_${search_strategies}_lp${len_penalty}_t${temperature}_${beam_size}_bhs${beam_head_second}_bss${beam_size_second}_b${batch_size}_${data_folder}.txt

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
torchrun \
      --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
      NAG2G/validate.py $data_path --user-dir ./NAG2G \
      --valid-subset test \
      --task ${task} --loss G2G --arch NAG2G_G2G --encoder-type ${encoder_type} \
      --seed $seed \
      --infer_step \
      --results-path $results_path \
      --path $path \
      --num-workers ${num_workers} --ddp-backend=no_c10d \
      --required-batch-size-multiple 1 \
      --search_strategies ${search_strategies} \
      --beam-size ${beam_size} --len-penalty ${len_penalty} --temperature ${temperature} \
      --beam-size-second ${beam_size_second} --beam-head-second ${beam_head_second} \
      --infer_save_name ${infer_save_name} \
      --batch-size $batch_size \
      --data-buffer-size ${batch_size} --fixed-validation-seed 11 --batch-size-valid ${batch_size} \
      --config_file $config_file

cd NAG2G
new_filename=$(echo "$infer_save_name" | sed 's/.txt/_{}.txt/')
python G2G_cal.py ${results_path}/${new_filename} ${beam_size}
cd -
