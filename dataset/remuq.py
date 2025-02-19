import os
import json
from tqdm import tqdm
from collections import defaultdict
from dataset.base import BaseRetrievalDataset, load_jsonl, convert_list2dict
from dataset.vqa_ret import parse_pairs


class ReMuQRetrievalDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_lineidx, img_tsv, query_tokenizer, doc_tokenizer, img_processor):
        super().__init__("", img_processor, query_tokenizer, doc_tokenizer, False)
        
        samples = parse_pairs(data_path)
        with open(img_lineidx, "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        
        self.fp = open(img_tsv, "r")
        
        for sample in tqdm(samples):
            self.process_sample(sample)
        del samples
        self.fp.close()
    
    def process_sample(self, sample):
        question = sample["question"]        
        passage = sample["document"].strip()
        image_id = sample["image"]
        self.fp.seek(self.lineidx[int(image_id) % 10000000])
        _, img_base64 = self.fp.readline().strip().split("\t")
        
        passage = self.tensorize_doc(passage)

        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": passage,
            "I_q": img_base64, "I_d": [] # "A": answer, 
        })