import os
import json
import linecache
import re, jsonlines
from collections import defaultdict
from dataset.base import BaseRetrievalDataset, load_jsonl, convert_list2dict
from retrievers.colbert.data import Collection
from dataset.vqa_utils import VQA
import torch
from tqdm import tqdm
from PIL import Image
import random
from dataset.vqa_ret import parse_pairs as parse_pairs_for_gs

def imageid2path(img_id, split="train|val"):
    assert split in ["train", "val"]
    
    img_id = str(img_id)
    prefix = (12 - len(img_id))*"0"
    image_path = f"coco/{split}2014/COCO_{split}2014_{prefix}{img_id}.jpg"
    
    return image_path
    
def get_collection(filename, return_ids=False):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    passage_ids = []
    passages = []            
    for line in lines:
        entry = json.loads(line.strip())
        passage_ids.append(entry["id"])
        passages.append(entry['text'])
        
    print(f"Loaded {len(passage_ids)} passages.")
    
    collection = Collection(data=passages)
    
    if return_ids:
        return collection, passage_ids
    
    return collection


def parse_pairs(filename):
    if "train" in filename:
        split = "train"
    elif "val" in filename or "test" in filename:
        split = "val"
    
    with open(filename, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        entry = json.loads(line.strip())
        question_id = int(entry["question_id"])
        image_id = entry["image_id"]
        question = entry["question"]
        # answers = entry["answers"]
        pos_passage = entry["pos_passage"]["passage"]
        # neg_passage = entry["neg_passage"]["passage"]
        
        image_path = imageid2path(image_id, split=split)
        
        data.append({
            "question": question, "document": pos_passage,
            "image": image_path, "q_id": question_id
        })
    
    return data

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, test, prompt, image_dir, image_processor=None, retrieved=False):
        if test.split(".")[-1]=="jsonl":
            samples = []
            with jsonlines.open(test, "r") as f:
                for line in f.iter():
                    samples.append(line)
            self.test = samples
        else:
            self.test = json.load(open(test))
            
        self.prompt = prompt
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.retrieved = retrieved
        
    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        image, question, question_id, annotation = data['image'], data['question'], data['question_id'], data.get('answer', None)
        
        if self.retrieved:
            prompt = []
            relevances = []
            if self.image_processor is not None:
                for context, R in zip(data['passages'], data["retrieval_score"]):
                    if len(context.split(" ")) > 300:
                        _c = context.split(" ")[:300]
                        context = " ".join(_c)
                    prompt.append(self.prompt.format(context, question))
                    relevances.append(R)
                image_tensor = image_tensor.expand(len(prompt), -1,-1,-1)
            else:
                ctx = []
                for context in data['passages']:
                    if len(context.split(" ")) > 200:
                        _c = context.split(" ")[:200]
                        context = " ".join(_c)
                    ctx.append(context)
                
                prompt = [f"Passage #{i+1} Text: {c}" for i, c in enumerate(ctx)]
                prompt = "\n".join(prompt)
                prompt = self.prompt.format(prompt, question)
        else:
            prompt = self.prompt.format(question)
        
        return_dict = {
            'question': prompt,
            'question_id': question_id,
            'annotation': annotation,
        }
        
        image_path = os.path.join(self.image_dir, os.path.basename(image)) 
        if self.image_processor is not None:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_processor(image)
            return_dict.update({"image_tensor": image_tensor})
        else:
            return_dict.update({"image_path": image_path})
        
        if self.retrieved:
            return_dict.update({"relevance_score": relevances})
        
        return return_dict

class OKVQARetrievalDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        samples = parse_pairs(data_path)
        
        # Caching
        if os.path.exists(data_path[:-3]+"pt"):
            print("Caching..")
            self.data = torch.load(data_path[:-3]+"pt")
        else:
            for sample in tqdm(samples):
                self.process_sample(sample)
            del samples
            
            torch.save(self.data, data_path[:-3]+"pt")
    
    def process_sample(self, sample):
        question = sample["question"]
        passage = sample["document"].strip()
        image_path = sample["image"]
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)
        
        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": self.tensorize_doc(passage),
            "I_q": q_image_path, "I_d": [], # "A": answer, 
        })

class OKVQAGoogleSearchDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, nways=64, caption=False, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        self.nways = nways
        self.caption = caption
        samples = parse_pairs_for_gs(data_path, random_choice=True, caption=caption)

        for sample in tqdm(samples):
            self.process_sample(sample)
        del samples
    
    def process_sample(self, sample):
        question = sample["question"]        
        passage = sample["document"].strip()
        image_path = sample["image"]
        _, sp, img_id = image_path.split("_")
        image_path = imageid2path(img_id, sp.split("20")[0])
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)
        
        passage = self.tensorize_doc(passage)

        if self.nways > 1:
            # if not "negs" in sample or len(sample["negs"]) < self.nways-1:
            #     comp = random.choice(self.data)
            #     if not "negs" in sample:
            #         hard_negs = comp["negs"]
            #     else:
            #         hard_negs = sample["negs"] + random.sample(comp["negs"], self.nways-1-len(sample["negs"]))
            # else:
            if "negs" in sample:
                hard_negs = sample["negs"]
            else:
                print(sample)

        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": passage,
            "I_q": q_image_path, "I_d": [] # "A": answer, 
        })

        if self.nways > 1:
            self.data[-1]["negs"] = hard_negs
        

class DynamicEval():
    def __init__(self, ann_file, ques_file, passage_id_to_line_id_file, all_blocks_file):
        
        with open(passage_id_to_line_id_file) as fin:
            self.passage_id_to_line_id = json.load(fin)
            
        self.vqa = VQA(ann_file, ques_file)
        self.all_blocks_file = all_blocks_file
            
    
    def get_answers(self, question_id):
        ann = self.vqa.loadQA(question_id)
        qa = self.vqa.returnQA(ann)[0]
        answers = set(answer.lower() for answer in qa['answers'].values() if answer)
        return answers
    
    
    def get_passage(self, passage_id):
        passage_line = linecache.getline(
            self.all_blocks_file, self.passage_id_to_line_id[passage_id])
        passage_dict = json.loads(passage_line)
        passage = passage_dict['text']
        assert passage_id == passage_dict['id']

        return passage
    
    
    def has_answers(self, answers, passage):
        passage = passage.lower()
        for answer in answers:
            answer = answer.strip().lower()
            # "\b" matches word boundaries.
            # answer_starts = [match.start() for match in re.finditer(
            #     r'\b{}\b'.format(answer.lower()), passage)]
            if re.search(r'\b{}\b'.format(answer), passage):
                return True
        return False
    
    
    def gen_qrels(self, question_ids, I, retrieved_id_to_passage_id):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in tqdm(zip(question_ids, I), total=len(question_ids)):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}

            for retrieved_id in retrieved_ids:
                passage_id = retrieved_id_to_passage_id[retrieved_id]
                answers = self.get_answers(int(question_id))
                passage = self.get_passage(passage_id)

                if self.has_answers(answers, passage):
                    qrels[str(question_id)][passage_id] = 1

        return qrels
    
def prepare_RAG_data(data_path):
    with open(data_path, "r") as fin:
        data = json.load(fin)
    
    print("#ORI:", len(data))
    rag_data = []
    for sample in data:
        image_file = sample["img_id"]+".jpg"
        for ctx in sample['ctxs']:
            doc = ctx["text"]
            
            # check answer
            answer = None
            gt_answers = list(sample["answers"].keys())
            for ans in gt_answers:
                if ans in doc and not ans in ["unknown", "unanswerable"]:
                    answer = ans
                    break
            
            if answer is None:
                answer = gt_answers[0]
                
            rag_data.append({
                "image": image_file,
                "question_id": f"{sample['question_id']}_{ctx['id']}",
                "question": sample["question"],
                "context": doc,
                "answer": answer
            })
    
    print("# DATA:", len(rag_data))
    with open(data_path[:-5]+"_rag.json", "w") as fout:
        json.dump(rag_data, fout, ensure_ascii=False, indent=2)
        

if __name__=="__main__":
    prepare_RAG_data("data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/retriever_train.json")