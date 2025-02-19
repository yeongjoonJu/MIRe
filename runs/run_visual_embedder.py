import argparse, os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from retrievers.mire import build_vision_tower
from dataset.base import ImageDataset
from transformers import CLIPImageProcessor
from dataset.infoseek import kb_id2img_path
from tqdm import tqdm
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--data_path", type=str, required=True, help="Path to a file with image path list")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vision_tower(args.model_name)
    model = nn.DataParallel(model)  # Use DataParallel
    model.to(device)
    model.eval()
    
    """
    Process data
    """
    with open(args.data_path, "r") as f:
        data = json.load(f)
        
    if 'image' in data[0]:
        img_list = [sample["image"] for sample in data]
    elif 'image_id' in data[0]:
        img_list = [kb_id2img_path(sample['image_id']) for sample in data]
    else:
        raise NotImplementedError
    
    img_list = list(set(img_list))
    
    ## checking exists
    target_list = []
    for img in tqdm(img_list):
        if not os.path.exists(os.path.join(args.image_dir, img[:-3]+"npy")):
            target_list.append(img)
    
    img_processor = CLIPImageProcessor.from_pretrained(args.model_name)
    img_dataset = ImageDataset(target_list, args.image_dir, img_processor=img_processor)
    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        for batch in tqdm(img_dataloader):
            images, paths = batch
            images = images.to(device)
            
            outputs = model(images)
            embeddings = outputs.cpu().numpy()
            
            for emb, path in zip(embeddings, paths):
                path = path[:-3] + "npy"
                # if not os.path.exists(path):
                np.save(path, emb)