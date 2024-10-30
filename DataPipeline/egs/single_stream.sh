#!/bin/bash
# This script transforms a mono-channel audio-only dataset into single stream codec codes.
# 
# Usage:
#   ./script_name.sh --db-root <path> --processed-root <path> [--wav-scp <path>] [--name <name>] [-h]
#
# Options:
#   --db-root         Path to the root directory of the dataset.
#                     All .wav files under this directory will be processed if --wav-scp is not specified.
#   --processed-root  Path to the root directory of the processed audio / metadata.
#   --wav-scp             Path to the wav.scp file. All paths in the wav.scp file should be absolute path under --db-root.
#   --name            Name of the dataset. This will be used as the name of the directory under --processed-root.
#   -h, --help        Show this help message and exit.


stage=1
stop_stage=4
thread_per_gpu=12

export CUDA_VISIBLE_DEVICES=1
ngpu=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Available GPUs: $ngpu"
num_workers=$(($ngpu * $thread_per_gpu))

# wav_scp=""
db_root="/mnt/Corpus/Speech/OpenData/LJSpeech"
processed_root="/mnt/users/hccl.local/jkzhao/data/LJSpeech_processed"
llm_ckpt_dir="/mnt/users/hccl.local/jkzhao/ckpts/meta-llama/Meta-Llama-3.1-8B-Instruct"
# llm_ckpt_dir="/mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
name="ljspeech"


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --db-root) db_root="$2"; shift ;;
        --processed-root) processed_root="$2"; shift ;;
        --wav-scp) wav_scp="$2"; shift ;;
        --name) name="$2"; shift ;;

        -h|--help) echo "Usage: $0 --db-root <path> --processed-root <path> [--wav-scp <path>] [--name <name>]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

work_dir=$(dirname $(dirname $(realpath $0)))
echo "Work directory: $work_dir"
export PYTHONPATH=$work_dir:$PYTHONPATH
log_dir=$work_dir/data/$name/log
mkdir -p $log_dir

# conda activate moshi-data
# Prepare wav.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -z "$scp_dir" ]; then
        echo "Prepare wav.scp"
        wav_scp=$log_dir/wav.scp
        find "${db_root}" -type f -name "*.wav" | while read -r wav_file; do
            id=$(basename $wav_file .wav)
            echo "$id $wav_file" >> $wav_scp
        done
    fi
fi

# Split the data for $num_workers GPUs
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $num_workers GPUs"
    mkdir -p $log_dir/${num_workers}splits
    # extra shuf to ensure balance across GPUs
    # So the generated data cannot be reproduced due to the shuffle randomness
    cat $log_dir/wav.scp | shuf >  $log_dir/wav.scp.shuf
    split_scp=
    for n in `seq 1 $num_workers`; do
        split_scp="$split_scp $log_dir/${num_workers}splits/wav.${n}.scp"
    done
    $work_dir/utils/split_scp.pl $log_dir/wav.scp.shuf $split_scp
fi

# Data Preprocessing
# We use relative path in wav.scp
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # ASR
    # This will yield wav_asr.JOB.scp and utt2json
    echo "ASR with whisperX"
    num_files=$(cat $log_dir/wav.scp | wc -l)
    start_time=$(date +%s)

    $work_dir/utils/run.pl JOB=1:$num_workers  $log_dir/${num_workers}splits/log/asr.JOB.log \
    python $work_dir/local/asr_whisperx.py \
        --rank JOB \
        --input-file $log_dir/${num_workers}splits/wav.JOB.scp \
        --audio-input-dir $db_root \
        --output-file $log_dir/${num_workers}splits/wav_asr.JOB.scp \
        --metadata-output-dir "${processed_root}/metadata"

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "ASR processing time: $elapsed_time seconds"
    echo "ASR processing speed: $(echo "scale=2; $elapsed_time / $num_files * $ngpu" | bc) s/files per GPU"
    echo "ASR processing speed: $(echo "scale=2; $elapsed_time / $num_files * $num_workers" | bc) s/files per thread"
fi

# conda activate open-moshi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare text and audio sequence"
    mkdir -p ${processed_root}/codecs
    $work_dir/utils/run.pl JOB=1:$num_workers  $log_dir/${num_workers}splits/log/tokenize_dump.JOB.log \
    python3 $work_dir/local/tokenize_monochannel.py \
        --input-text-file $log_dir/${num_workers}splits/utt2json.JOB \
        --output-file ${processed_root}/codecs/${part}/${num_workers}splits/codec.JOB.pt \
        --root-dir $processed_root \
        --rank JOB \
        --llm-ckpt-dir $llm_ckpt_dir || exit 1;
fi
        # --input-audio-file $log_dir/${num_workers}splits/wav.JOB.scp \