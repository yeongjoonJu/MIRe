import torch
from torch.amp import autocast
import wandb
import retrievers.utils as utils

# def log_results_training_step(train_stats):
#     log_stats = {}
#     if train_stats:
#         log_stats.update({f"train_step_{k}": v for k, v in train_stats.items()})
        
#     return log_stats

def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "\rTrain Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    is_bf16 = config.model.short_name=="sent_mire"
    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    print()
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # autocast for mixed precision
        dtype = torch.bfloat16 if config.model.short_name=="sent_mire" else torch.float32
        with autocast('cuda', dtype=dtype):
            outputs = model(
                batch["T_q_ids"], batch["T_q_masks"], batch["I_q"], batch["I_q_masks"],
                doc_ids=batch["T_d_ids"], doc_attention_mask=batch["T_d_masks"],
                doc_images=batch["I_d"], doc_image_mask=batch["I_d_masks"],
                query_image_embs=batch["I_q_emb"], doc_image_embs=batch["I_d_emb"],
                return_loss=True,
                # gather_embeddings=True,
            )
        loss = outputs["loss"]
        inbatch_accuracy = outputs["accuracy"]

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        if is_bf16:
            loss.backward()
        else:
            # Use scaler for backward
            scaler.scale(loss).backward()
        
        if utils.is_main_process():
            wandb.log({"loss per training step": loss.item()})
        
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1
            
            if is_bf16:
                optimizer.step()
            else:
                # optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            

            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # Save model states
    saved_state = model_without_ddp.state_dict()

    print("Cleared model queue states.")

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # autocast for mixed precision
        dtype = torch.bfloat16 if config.model.short_name=="modernxknow" else torch.float32
        with autocast('cuda', dtype=dtype):
            outputs = model(
                batch["T_q_ids"], batch["T_q_masks"], batch["I_q"], batch["I_q_masks"],
                doc_ids=batch["T_d_ids"], doc_attention_mask=batch["T_d_masks"],
                doc_images=batch["I_d"], doc_image_mask=batch["I_d_masks"],
                query_image_embs=batch["I_q_emb"], doc_image_embs=batch["I_d_emb"],
                return_loss=True, return_ib_acc=True
                # gather_embeddings=True,
            )
            loss = outputs["loss"]
            inbatch_accuracy = outputs["accuracy"]

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # Restore model states from the saved variables
    model_without_ddp.load_state_dict(saved_state)
    print("Restored model queue states and model states from the saved variables.")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}