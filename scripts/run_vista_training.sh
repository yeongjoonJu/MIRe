CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vista.py \
    --encoder BAAI/bge-base-en-v1.5 \
    --resume_path ckpts/vista/Visualized_base_en_v1.5.pth \
    --data_path data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/retriever_train_w_negs.json \
    --img_dir data/vid2r/images \
    --batch_size 64 \
    --epochs 5