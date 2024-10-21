#!/bin/bash
# megatron to transformers: You need to copy the tokenizer files into the save_path
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-hf2mg-test-2-2/release/ ../../llama_mg2hf 1 1 llama-7b 1 true
# transformers to megatron
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-7b-hf ../../llama-hf2mg 1 1 llama-7b 1 false
set -e
START_TIME=$SECONDS

MEGATRON_PATCH_PATH=/cq_1/share_1603164/user/kaixinma/Pai-Megatron-Patch
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240424

SOURCE_CKPT_PATH=your_path/llama-3/hf/Meta-Llama-3-70B-Instruct

TP=8
PP=1
MN=llama2-70b
EXTRA_VOCAB_SIZE=0
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

TARGET_CKPT_PATH=your_path/idefics2-llama3-70b-tp-${TP}-pp-${PP}

echo ${MEGATRON_PATH}
python idefics2_hf2mg_llama3.py \
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
