# Standard library
import argparse
import logging
import os
import random
import math
import multiprocessing

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb
from retrievers.mire import MIRe, SentMIRe
from retrievers.preflmr import PreFLMR

from dataset import get_dataloader, ViD2RDataset, VL_ICT
from dataset.okvqa import OKVQAGoogleSearchDataset, OKVQARetrievalDataset
from dataset.aokvqa import AOKVQADataset
from dataset.m2kr import M2KRDataset
from dataset.remuq import ReMuQRetrievalDataset
import json

from transformers import CLIPImageProcessor, AutoConfig, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from retrievers.engine import train_one_epoch, eval_engine
from retrievers.colbert import QueryTokenizer, DocTokenizer, ColBERTConfig
import retrievers.utils as utils

# Set up logger
logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]


def create_optimizer(gain_or_bias_params, rest_params, config, args, t5_params=None):
    params = [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": 0.2},
    ]
    # TODO: if options are modified, remove args
    if t5_params is not None:
        params.append({"params": t5_params, "weight_decay": 0.2, "lr": config.trainer_config.t5_learning_rate},)
    
    return optim.AdamW(
        params,
        lr=config.trainer_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    ckpt_config = config.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    global_step, total_loss, best_inbatch_accuracy = 0, 0.0, 0.0 # TODO: global_step is not used.
    best_epoch = 0
    model.zero_grad()

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        # if is_distributed_mode:
        #     train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )

        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            val_status = eval_engine(model_without_ddp, model, val_loader, gpu_id, config)
            try:
                inbatch_accuracy = float(val_status["inbatch_accuracy"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['inbatch_accuracy']}'")
                inbatch_accuracy = 100.0
            # Note: still save the model even if the in-batch accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            if inbatch_accuracy >= best_inbatch_accuracy:
                best_inbatch_accuracy = inbatch_accuracy
                best_epoch = epoch
            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch)

        if utils.is_main_process():
            if config.wandb_config.enabled:
                wandb.log(log_stats)

        dist.barrier()  # Wait for the master process to finish writing the log file
        torch.cuda.empty_cache()
        

def main(config, args):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    # TODO: is this necessary?
    cudnn.benchmark = True

    # Initialize and load model
    print("Creating model...")
    
    # Set model configuration
    colbert_ckpt = config.model.colbert_checkpoint
    model_cfg = AutoConfig.from_pretrained(colbert_ckpt)
    model_cfg.update(config.model)
    
    # Set colbert configuration
    try:
        col_config = ColBERTConfig.load_from_checkpoint(colbert_ckpt)
        col_config.nway = config.data_config.nways
    except:
        col_config = ColBERTConfig()
        col_config.checkpoint = colbert_ckpt
        col_config.use_ib_negatives = True
        col_config.kmeans_niters = 20
        col_config.nbits = 2
        col_config.nway = config.data_config.nways
    
    if config.model.short_name=='mire':
        model = MIRe(model_cfg, colbert_config=col_config)
    elif config.model.short_name=="preflmr":
        model = PreFLMR(model_cfg, colbert_config=col_config)
    elif config.model.short_name=="sent_mire":
        model = SentMIRe(model_cfg, colbert_config=col_config)
        col_config.query_token = model.query_prefix
        col_config.query_token_id = model.query_prefix
        col_config.doc_token = model.document_prefix
        col_config.doc_token_id = model.document_prefix
        
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print("#Params:", count_parameters(model))
    
    if config.trainer_config.pretrained_checkpoint:
        print("Load checkpoint: ", config.trainer_config.pretrained_checkpoint)
        model.load_state_dict(torch.load(config.trainer_config.pretrained_checkpoint)["model"])

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.trainer_config.learning_rate,
        weight_decay=config.trainer_config.weight_decay,
        eps=1e-8
    )
        
    scaler = GradScaler()  # Initialize the GradScaler

    # If resume training, load the checkpoint
    ckpt_config = config.ckpt_config
    model_cfg.save_pretrained(ckpt_config.ckpt_dir)
    col_config.save_for_checkpoint(ckpt_config.ckpt_dir)
    model.raw_tokenizer.save_pretrained(ckpt_config.ckpt_dir)
    with open(f"{ckpt_config.ckpt_dir}/train_hyperparams.json", "w") as fout:
        json.dump(dict(config.trainer_config), fout, ensure_ascii=False, indent=2)
        
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading model checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(f"cuda:{config.dist_config.gpu_id}"))
        model = model.to(config.dist_config.gpu_id)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        
    # Move model to GPUs
    model.train()

    frozen = config.trainer_config.frozen
    linear_frozen = config.get("linear_frozen", frozen)
    model.freeze_parameters(frozen, linear_frozen)
    model = model.to(config.dist_config.gpu_id)
    
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id], find_unused_parameters=True)
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    
    query_tokenizer = QueryTokenizer(col_config)
    doc_tokenizer = DocTokenizer(col_config)
    img_processor = CLIPImageProcessor.from_pretrained(model_cfg.vision_model_name)
    if config.model.pretraining:
        query_tokenizer.query_maxlen = 40
        doc_tokenizer.doc_maxlen = 400
    else:
        query_tokenizer.query_maxlen = 32
        doc_tokenizer.doc_maxlen = 180

    MASK_TOKEN_ID = doc_tokenizer.tok.mask_token_id
    PAD_TOKEN_ID  = doc_tokenizer.tok.pad_token_id

    data_cfg = config.data_config
        
    if data_cfg.dataset_name=="okvqa_gs":
        src = data_cfg.okvqa
        train_dataset = OKVQAGoogleSearchDataset(
                        src.data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor, nways=col_config.nway,
                        img_cached=src.image_cached)
        valid_dataset = OKVQAGoogleSearchDataset(
                        src.valid_data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor, nways=1,
                        img_cached=src.image_cached)
        
    elif data_cfg.dataset_name=="okvqa":
        src = data_cfg.okvqa
        train_dataset = OKVQARetrievalDataset(
                        src.data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
        valid_dataset = OKVQARetrievalDataset(
                        src.valid_data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
        
    elif data_cfg.dataset_name=="aokvqa":
        src = data_cfg.aokvqa
        train_val_dataset = AOKVQADataset(
                        src.data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
        val_samples = src.valid_samples
        train_dataset, valid_dataset = random_split(train_val_dataset, [len(train_val_dataset)-val_samples, val_samples])
        
    elif data_cfg.dataset_name=="vid2r":
        src = config.data_config.vid2r
        train_val_dataset = ViD2RDataset(src.data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
            
        val_samples = src.valid_samples
        train_dataset, valid_dataset = random_split(train_val_dataset, [len(train_val_dataset)-val_samples, val_samples])
        
    elif data_cfg.dataset_name=="vl_ict":
        src = config.data_config.vl_ict
        train_dataset = VL_ICT(src.data_path, src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
        valid_dataset = VL_ICT(src.data_path.replace("train", "val"), src.img_dir,
                        query_tokenizer, doc_tokenizer, img_processor,
                        img_cached=src.image_cached)
        
    elif data_cfg.dataset_name=="remuq":
        src = config.data_config.remuq
        train_dataset = ReMuQRetrievalDataset(src.data_path, src.img_lineidx, src.img_tsv,
                        query_tokenizer, doc_tokenizer, img_processor,)
        valid_dataset = train_dataset

    elif data_cfg.dataset_name in ["infoseek", "evqa"]:
        src = data_cfg.m2kr
        if data_cfg.dataset_name=="infoseek":
            val_samples = src.valid_samples
            train_val_dataset = M2KRDataset(src.subset, src.img_dir,
                            query_tokenizer, doc_tokenizer, img_processor, split="train")
            train_dataset, valid_dataset = random_split(train_val_dataset, [len(train_val_dataset)-val_samples, val_samples])
        else:
            train_dataset = M2KRDataset(src.subset, src.img_dir,
                            query_tokenizer, doc_tokenizer, img_processor, split="train")
            valid_dataset = M2KRDataset(src.subset, src.img_dir,
                            query_tokenizer, doc_tokenizer, img_processor, split="valid")

    
    if args.caching:
        return

    train_loader = get_dataloader(train_dataset, shuffle=True, batch_size=config.dataloader_config.train_batch_size,
                                num_workers=config.dataloader_config.num_workers, MASK_TOKEN_ID=MASK_TOKEN_ID, PAD_TOKEN_ID=PAD_TOKEN_ID)
    valid_loader = get_dataloader(valid_dataset, shuffle=False, batch_size=config.dataloader_config.valid_batch_size,
                                num_workers=config.dataloader_config.num_workers, MASK_TOKEN_ID=MASK_TOKEN_ID, PAD_TOKEN_ID=PAD_TOKEN_ID)
    

    # Initializing the scheduler
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    
    if config.model.pretraining:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.trainer_config.warmup_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.trainer_config.warmup_steps)
    

    epoch = 0
    if ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1

    # Training loop
    dist.barrier()
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )
    

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--caching", action="store_true", help="Whether to cache the dataset.")

    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    # Set up wandb
    if config.wandb_config.enabled and utils.is_main_process():
        load_dotenv()  # Load .env and get WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY
        wandb_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Set up logger
    if utils.is_main_process():
        # logger_out_dir = os.path.join(config.mbeir_dir, config.logger_config.logger_out_dir)
        logger_out_dir = config.logger_config.logger_out_dir
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        if not os.path.exists(logger_out_dir):
            os.makedirs(logger_out_dir, exist_ok=True)
        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info(config)

    main(config, args)

    # Close wandb
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()