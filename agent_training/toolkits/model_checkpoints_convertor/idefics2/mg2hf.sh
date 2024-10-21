# cot
MODEL_PATH=idefics-new-cot-tn5-sr0.25-in4-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/hd_output/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot wd
MODEL_PATH=idefics-new-cot-wd-tn5-sr0.25-in4-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot wd in2
MODEL_PATH=idefics-new-cot-wd-in2-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1024-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot wd in6
MODEL_PATH=idefics-new-cot-wd-in6-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1280-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot wd in8
MODEL_PATH=idefics-new-cot-wd-in8-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1536-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}


# cot nd
MODEL_PATH=idefics-new-cot-nd-tn5-sr0.25-in4-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot nd in2
MODEL_PATH=idefics-new-cot-nd-in2-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot nd in6
MODEL_PATH=idefics-new-cot-nd-in6-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1280-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# cot nd in8
MODEL_PATH=idefics-new-cot-nd-in8-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1536-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}


# std
MODEL_PATH=idefics-new-hd-wd-tn5-sr0.25-in4-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 4stp 4in wd
MODEL_PATH=idefics-new-cot-wd-stp4-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1024-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 12stp 4in wd
MODEL_PATH=idefics-new-cot-wd-stp12-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1280-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 16stp 4in wd
MODEL_PATH=idefics-new-cot-wd-stp16-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1536-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 4stp 4in nd
MODEL_PATH=idefics-new-cot-nd-stp4-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1024-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 12stp 4in nd
MODEL_PATH=idefics-new-cot-nd-stp12-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1280-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 16stp 4in nd
MODEL_PATH=idefics-new-cot-nd-stp16-in4-tn5-sr0.25-lr-2e-5-bs-8-seqlen-1536-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0003660
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# trial number
# 1tn 4in nd
MODEL_PATH=idefics-new-cot-nd-in4-tn1-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0000731
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 2tn 4in nd
MODEL_PATH=idefics-new-cot-nd-in4-tn2-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0001462
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 3tn 4in nd
MODEL_PATH=idefics-new-cot-nd-in4-tn3-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0002195
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 4tn 4in nd
MODEL_PATH=idefics-new-cot-nd-in4-tn4-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0002925
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}


# 1tn 4in wd
MODEL_PATH=idefics-new-cot-wd-in4-tn1-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0000731
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 2tn 4in wd
MODEL_PATH=idefics-new-cot-wd-in4-tn2-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0001462
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 3tn 4in wd
MODEL_PATH=idefics-new-cot-wd-in4-tn3-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0002195
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}

# 4tn 4in wd
MODEL_PATH=idefics-new-cot-wd-in4-tn4-sr0.25-lr-2e-5-bs-8-seqlen-1152-pr-bf16-tp-8-pp-1-ac-full-do-false-sp-false-warmup66
IN_DIR=iter_0002925
OUT_DIR=hf_ckp
sh mg2hf_base_idefics_instruct.sh ${MODEL_PATH} ${IN_DIR} ${OUT_DIR}
rsync -av --ignore-existing your_path/huggingface_models/idefics2-8b/* your_path/pretrain_data/embodied_model/checkpoint/${MODEL_PATH}/${OUT_DIR}
