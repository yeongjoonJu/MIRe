import os
import torch
from dataset.eval_loaders import M2KRLoader
from dataset.base import BaseRetrievalDataset
from tqdm import tqdm

class M2KRDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, split="train"):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, False)
        
        # Caching
        if os.path.exists(data_path+"_m2kr_"+split+".pt"):
            print("Caching..")
            self.data = torch.load(data_path+"_m2kr_"+split+".pt")
        else:
            loader = M2KRLoader(None)
            samples = loader.parse_pairs(data_path, img_dir, split=split)
            for sample in tqdm(samples, desc="Processing samples"):
                self.process_sample(sample)
            del samples

            torch.save(self.data, data_path+"_m2kr_"+split+".pt")
            print("Save cache!")
        
    def process_sample(self, sample):
        question = sample["question"]
        passage = sample["document"].strip()
        image_path = sample["image"]
        passage = self.tensorize_doc(passage)

        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": passage,
            "I_q": image_path, "I_d": [] # "A": answer, 
        })