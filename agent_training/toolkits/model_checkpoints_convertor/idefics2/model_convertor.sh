#!/bin/bash
# megatron to transformers: You need to copy the tokenizer files into the save_path
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-hf2mg-test-2-2/release/ ../../llama_mg2hf 1 1 llama-7b 1 true
# transformers to megatron
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-7b-hf ../../llama-hf2mg 1 1 llama-7b 1 false
set -e
START_TIME=$SECONDS

MEGATRON_PATCH_PATH=your_path/Pai-Megatron-Patch
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240126

SOURCE_CKPT_PATH=your_path/huggingface_models/idefics2-8b

TP=4
PP=1
MN=mistral-7b
EXTRA_VOCAB_SIZE=125
mg2hf=false
dtype=bf16

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

TARGET_CKPT_PATH=your_path/megatron_models/idefics2-8b-instruct/idefics2-mistral-megatron-tp-${TP}-pp-${PP}


python idefics2_hf2mg.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype ${dtype} \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
