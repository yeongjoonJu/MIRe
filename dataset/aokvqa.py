import os
import json
from tqdm import tqdm
from dataset.base import BaseRetrievalDataset
import torch

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(image_id, coco_dir):
    return os.path.join(coco_dir, f"{image_id:012}.jpg")

def parse_pairs(filename, image_dir=None):
    samples = json.load(open(filename))
    data = []
    passages = {}
    for sample in samples:
        question = sample["question"]
        q_id = sample["question_id"]
        passage = " ".join(sample["rationales"])
        if image_dir is not None:
            image_path = get_coco_path(sample["image_id"], image_dir)
            data.append({
                "question": question, "image": image_path, "q_id": q_id, "answers": sample["direct_answers"]
            })
        else:
            data.append({
                "question": question, "q_id": q_id, "answers": sample["direct_answers"]
            })
        passages[q_id] = passage
    
    return data, passages

class AOKVQADataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        samples = json.load(open(data_path))
        
        # Caching
        # if os.path.exists(data_path[:-4]+"pt"):
        #     print("Caching..")
        #     self.data = torch.load(data_path[:-4]+"pt")
        # else:
        for sample in tqdm(samples):
            self.process_sample(sample)
        del samples
        
        # torch.save(self.data, data_path[:-4]+"pt")
    
    def process_sample(self, sample):
        question = sample["question"]        
        passage = " ".join(sample["rationales"])
        image_path = get_coco_path(sample["image_id"], self.img_dir)
        if self.img_cached:
            q_image_path = image_path[:-3]+"npy"
        else:
            q_image_path = image_path
        
        passage = self.tensorize_doc(passage)

        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": passage,
            "I_q": q_image_path, "I_d": [] # "A": answer, 
        })