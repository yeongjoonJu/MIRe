import os, json
from retrievers.colbert.data import Collection
from dataset.base import BaseRetrievalDataset
import re

def parse_pairs(filename, image_dir):
    samples = json.load(open(filename))
    data = []
    for q_id, values in samples.items():
        question = values['question']
        image_path = os.path.join(image_dir, values['img_file'])
        passages = [re.sub(r"[\[\]]", "", p) for p in values["fact_surface"]]
        data.append({
            "question": question, "image": image_path, "q_id": q_id, "answers": values["answer"],
            "passage_ids": values['fact'], "passages": passages
        })
    
    return data
        
    
def get_collection(filename):
    triplets = json.load(open(filename))
    pid2passage = {}
    for k, v in triplets.items():
        pid2passage[k] = re.sub(r"[\[\]]", "", v['surface'])
    
    collection = Collection(data=list(pid2passage.values()))
    
    return collection, pid2passage