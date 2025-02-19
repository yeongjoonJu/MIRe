# # Update config
export WANDB_API_KEY=[WANDB_API_KEY]

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3 # <--- Change this to the CUDA devices you want to us
NPROC=4
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

MASTER_ADDR='localhost'
MASTER_PORT=29601


# OK-VQA Google Search
CONFIG_PATH=[CONFIGURE_FILE_PATH]
echo "CONFIG_PATH: $CONFIG_PATH"
python3 -m torch.distributed.run --master_port $MASTER_PORT --nproc_per_node=$NPROC train_retrieval.py --config_path "$CONFIG_PATH"