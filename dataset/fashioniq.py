import os
import torch
from dataset.base import BaseRetrievalDataset, load_jsonl, ImageCollection

def get_collection(filename, img_dir, return_ids=False):
    img_db = load_jsonl(filename)
    passages = []
    passage_ids = []
    for i in range(len(img_db)):
        name = os.path.basename(img_db[i]["image"])
        passages.append(os.path.join(img_dir, name))
        passage_ids.append(img_db[i]['content'])
    
    print(f"Loaded {len(img_db)} images.")

    collection = ImageCollection(data=passages)
    
    if return_ids:
        id2passage = {k: v for k, v in zip(passage_ids, passages)}
        return collection, passage_ids, id2passage
    
    return collection

def parse_pairs(filename, img_dir):
    samples = load_jsonl(filename)
    
    data = []
    for id, sample in enumerate(samples):
        question = sample["q_text"]
        q_img = os.path.basename(sample["q_img"])
        q_id = f"{q_img[:-4]}_{id}"
        image_path = os.path.join(img_dir, q_img)
        
        data.append({
            "question": question, #"document": passage,
            "image": image_path, "q_id": q_id, "golden_id": sample["positive"]
        })
        
    return data
