import json, os
import re, random
from tqdm import tqdm
# import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import torch
import numpy as np
from retrievers.colbert.data import Collection
from dataset.base import BaseRetrievalDataset, load_jsonl, convert_list2dict


def kb_id2img_path(kb_id):
    assert kb_id[0]=='Q'
    number = kb_id[1:]
    
    if len(number)==2:
        return f"Q{number}/Q{number}.jpg"
    
    group_id = number[:3]
    return f"Q{group_id}/Q{number}.jpg"

def get_collection(filename, return_ids=False):
    wiki_db = load_jsonl(filename)

    passages = []
    passage_ids = []
    for row in tqdm(wiki_db):
        passage_ids.append(row["wikidata_id"])
        title = row["wikipedia_title"]
        passage = row["wikipedia_content"]
        passage = f"title: {title} content: {passage}"
        passages.append(passage)

    print(f"Loaded {len(wiki_db)} passages.")

    collection = Collection(data=passages)

    if return_ids:
        id2passage = {k: v for k, v in zip(passage_ids, passages)}
        return collection, passage_ids, id2passage#, wiki_db
    
    return collection


def parse_pairs(filename):
    samples = load_jsonl(filename)
    # kb_mapping = convert_list2dict(kb_mapping, set_key="data_id")
    # wiki_db = convert_list2dict(wiki_db, set_key="wikidata_id")
    data = []
    for sample in samples:
        image_id = sample["image_id"]
        q_id = sample["data_id"].split("_")[-1]
        question = sample["question"]
        answers = sample["answer"]
        if "answer_eval" in sample:
            for ans in sample["answer_eval"]:
                if type(ans) is dict:
                    if "wikidata" in ans:
                        answers.append(str(ans["wikidata"]))
                    if "range" in ans:
                        answers.extend([str(a) for a in ans['range'][:2]])
                else:
                    answers.append(ans)
        answers = list(set(answers))
        
        # entity_id = kb_mapping[sample["data_id"]]["entity_id"]

        # title = wiki_db[entity_id]["wikipedia_title"].strip()
        # passage = wiki_db[entity_id]["wikipedia_content"].strip()
        # passage = f"title: {title} content: {passage}"

        image_path = f"{image_id}.JPEG"
        
        data.append({
            "question": question, #"document": passage,
            "image": image_path, "q_id": q_id, "answers": answers
        })

    return data


def filtering_cannot_answer(filename, kb_map_path, wiki_db_path):
    samples = load_jsonl(filename)
    kb_mapping = load_jsonl(kb_map_path)
    kb_mapping = convert_list2dict(kb_mapping, set_key="data_id")
    wiki_db = load_jsonl(wiki_db_path)
    wiki_db = convert_list2dict(wiki_db, set_key="wikidata_id")
    print("Complete preparation.")

    data = []
    for sample in tqdm(samples):
        answers = sample["answer"]
        if "answer_eval" in sample:
            for ans in sample["answer_eval"]:
                if type(ans) is dict:
                    if "wikidata" in ans:
                        answers.append(str(ans["wikidata"]))
                    if "range" in ans:
                        answers.extend([str(a) for a in ans['range'][:2]])
                else:
                    answers.append(ans)
        answers = list(set(answers))
        
        entity_id = kb_mapping[sample["data_id"]]["entity_id"]

        title = wiki_db[entity_id]["wikipedia_title"].strip()
        passage = wiki_db[entity_id]["wikipedia_content"].strip()
        passage = f"title: {title} content: {passage}"

        passage = passage.lower()

        contain = False
        for ans in answers:
            if re.search(r'\b{}\b'.format(ans.lower()), passage):
                contain = True
                break
        
        if contain:
            data.append(sample)

    return data
       

class InfoSeekDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer,
                 img_processor, kb_map_path, wiki_db_path, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer)
        
        # img2mapping = pd.read_csv(img_map_path, header=None)
        # img2mapping[1] = img2mapping[1].apply(lambda x: os.path.join(img_dir, str(x)))
        # self.img2mapping = pd.Series(img2mapping[1].values, index=img2mapping[0]).to_dict()

        if os.path.exists(data_path[:-5]+"pt"):
            print("Caching...")
            self.data = torch.load(data_path[:-5]+"pt")
        else:
            samples = load_jsonl(data_path)
            kb_mapping = load_jsonl(kb_map_path)
            self.kb_mapping = convert_list2dict(kb_mapping, set_key="data_id")
            wiki_db = load_jsonl(wiki_db_path)
            self.wiki_db = convert_list2dict(wiki_db, set_key="wikidata_id")
            self.wiki_img_dir = os.path.join(os.path.dirname(wiki_db_path), "wikipedia_images_full")

            for sample in tqdm(samples, desc="Processing samples"):
                self.process_sample(sample)
        
            del kb_mapping
            del self.kb_mapping
            del wiki_db
            del self.wiki_db

            for i in tqdm(range(len(self.data))):
                self.data[i]["T_q_ids"] = self.tensorize_query(self.data[i]["T_q_ids"])
                self.data[i]["T_d_ids"] = self.tensorize_doc(self.data[i]["T_d_ids"])

            torch.save(self.data, data_path[:-5]+"pt")

        
    def process_sample(self, sample):
        image_id = sample["image_id"]
        question = sample["question"]
        # answer = sample["answer"][0]
        entity_id = self.kb_mapping[sample["data_id"]]["entity_id"]
        title = self.wiki_db[entity_id]["wikipedia_title"]
        passage = self.wiki_db[entity_id]["wikipedia_content"]
        passage = f"title: {title.strip()} content: {passage.strip()}"
        passage = passage.strip()
        # q_image_path = self.img2mapping[image_id]
        q_image_path = os.path.join(self.img_dir, f"{image_id}.JPEG")
        if not os.path.exists(q_image_path):
            q_image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        # d_image_path = os.path.join(self.wiki_img_dir, kb_id2img_path(entity_id))
        # if not os.path.exists(d_image_path):
        #    d_image_path = ""
            
        # self.data.append({
        #     "T_q_ids": self.tensorize_query(question), "T_d_ids": self.tensorize_doc(passage),
        #     "I_q": q_image_path, "I_d": [], # "A": answer,
        # })
        self.data.append({
            "T_q_ids": question, "T_d_ids": passage,
            "I_q": q_image_path, "I_d": [], # "A": answer,
        })


class InfoSeekDataPool(BaseRetrievalDataset):
    def __init__(self, wiki_db_path, img_dir, img_processor, doc_tokenizer,):
        super().__init__(img_dir, img_processor, None, doc_tokenizer)
                
        wiki_db = load_jsonl(wiki_db_path)
        wiki_img_dir = os.path.join(os.path.dirname(wiki_db_path), "wikipedia_images_full")
        
        for row in tqdm(wiki_db):
            row_id = row["wikidata_id"]
            passage = row["wikipedia_content"]
            image_path = os.path.join(wiki_img_dir, kb_id2img_path(row_id))
            if not os.path.exists(image_path):
                image_path = ""
                
            self.data.append({
                "id": row_id, "T_d": passage, "I_d": image_path
            })
            
    def __getitem__(self, index):
        sample = {}
        for k, v in self.data[index].items():
            sample[k] = v
        
        if self.data[index]["I_d"]:
            I_d_path = self.data[index]["I_d"]
            if self.data[index]["I_d"][-3:]=="jpg":
                I_d = Image.open(I_d_path)
            else:
                with open(I_d_path, "r") as fp:
                    img_base64 = fp.readline().strip()
                    I_d = Image.open(BytesIO(base64.b64decode(img_base64)))
            sample["I_d"] = torch.tensor(self.img_processor(I_d)["pixel_values"][0])
            sample["I_d_masks"] = torch.tensor([1.])
        else:
            sample["I_d"] = torch.zeros((3, 224, 224))
            sample["I_d_masks"] = torch.tensor([0.])
            
        return sample