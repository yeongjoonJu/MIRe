#!/bin/bash

##################################
# Example script for evaluate_vista.py
##################################

PASSAGES="data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/okvqa_full_clean_corpus.csv"

CHECKPOINT="ckpts/vista/Visualized_BGE_ft.pth" #"ckpts/vista/Visualized_base_en_v1.5.pth"

# set GPU

CUDA_VISIBLE_DEVICES=0 python -m runs.evaluate_vista \
    --dataset_name okvqa_gs \
    --save_path experiments/eval_okvqa_gs_vista.memmap \
    --save_embedding true \
    --all_blocks_file "$PASSAGES" \
    --anno_file data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/retriever_test.json \
    --resume_path "$CHECKPOINT" \
    --image_dir data/vid2r/images/coco/val2014 \
    --encoder BAAI/bge-base-en-v1.5 \
    --batch_size 64 \
    --max_query_length 64 \
    --max_passage_length 384 \
    --index_factory "Flat" \
    --k 100


CUDA_VISIBLE_DEVICES=0 python3 -m runs.evaluate_vista \
    --dataset_name EVQA \
    --save_path experiments/eval_evqa_vista.memmap \
    --resume_path "$CHECKPOINT" \
    --image_dir data/evqa \
    --encoder BAAI/bge-base-en-v1.5 \
    --batch_size 64 \
    --max_query_length 64 \
    --max_passage_length 384 \
    --index_factory "Flat" \
    --k 100

PASSAGES=data/remuq/open_corpus_195837.csv
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.run_indexer --exp_name remuq --n_bits 2 --dataset_name remuq --all_blocks_file $PASSAGES
CUDA_VISIBLE_DEVICES=0 python3 -m runs.evaluate_vista \
    --dataset_name remuq \
    --save_path experiments/eval_remuq_vista_zero.json \
    --all_blocks_file $PASSAGES \
    --anno_file data/remuq/test.json \
    --image_dir data/remuq \
    --resume_path "$CHECKPOINT" \
    --encoder BAAI/bge-base-en-v1.5 \
    --batch_size 64 \
    --max_query_length 64 \
    --max_passage_length 384 \
    --index_factory "Flat" \
    --k 100