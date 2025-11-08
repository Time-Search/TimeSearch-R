DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="training"
OUTPUT_DIR=/path/to/your/local/experiments/$RUN_NAME/$DATETIME
mkdir $OUTPUT_DIR
export WANDB_PROJECT=TimeSearch-R-ColdStart
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
export DEBUG=true

export PYTHONPATH=".:$PYTHONPATH"
export SIGLIP_URL=grpc://10.136.101.146:51000
export LLM_AS_A_JUDGE_BASE=http://10.136.96.12:18901/v1

port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}" | awk -F',' '{print $2}')"
echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port in cmd: ${port_in_cmd}"

TRAIN_PATH=configs/dataset.yaml

VIDEO_ROOT=/path/to/your/local/datasets

MODEL_BASE=/path/to/your/local/SFT-checkpoint

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" \
    --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    time_r1/train.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --reward_func v7 \
    --prompt_template v4 \
    --tool_name_list seek_video_frames \
    --max_interaction_turns 8 \
    --max_prompt_length 18000 \
    --max_completion_length 16000 \
    --max_completion_length_per_turn 256 \
    --total_video_tokens 10240 \
    --max_frames 768 \
    --min_per_frame_tokens 12 \
    --max_per_frame_tokens 256 \
    --num_generations 8 \
    --scale_rewards false \
    --beta 0.005 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 1 \
    --dataloader_num_workers 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 200 \
    --save_only_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --shuffle_dataset true \
    --replay_buffer_type dapo \
    --use_counterfactual_reasoning true
