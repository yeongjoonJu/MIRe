import argparse
import json, os
from io import BytesIO
import base64
from collections import defaultdict
import pytrec_eval
import numpy as np
from tqdm import tqdm
from retrievers.colbert.data import Collection
from dataset.base import BaseDatasetEvalLoader

class OKVQALoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.okvqa import DynamicEval, get_collection, parse_pairs
        dynamic_eval = DynamicEval(
            "data/okvqa/mscoco_val2014_annotations.json",
            "data/okvqa/OpenEnded_mscoco_val2014_questions.json",
            passage_id_to_line_id_file="data/okvqa/passage_id_to_line_id.json",
            all_blocks_file=self.args.all_blocks_file
        )
        self.dynamic_eval = dynamic_eval

        pairs = parse_pairs(self.args.anno_file)
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            if self.args.image_dir is not None:
                self.images[d["q_id"]] = f"{self.args.image_dir}/{d['image']}"

        collection, passage_ids = get_collection(self.args.all_blocks_file, return_ids=True)
        self.collection = collection
        self.passage_ids = passage_ids

    def create_qrels(self, qids, I, passage_ids):
        # qrels 생성
        # dynamic_eval 이용
        return self.dynamic_eval.gen_qrels(qids, I, passage_ids)


class AOKVQALoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.aokvqa import parse_pairs
        pairs, pid2passage = parse_pairs(self.args.anno_file, image_dir=self.args.image_dir)
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            if self.args.image_dir is not None:
                self.images[d["q_id"]] = d['image']

        self.pid2passage = pid2passage
        self.passage_ids = list(pid2passage.keys())
        self.collection = Collection(data=list(pid2passage.values()))

    def create_qrels(self, qids, I, passage_ids):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                # AOKVQA에서 golden pid는 question_id와 동일한 경우만 relevant라고 가정
                if pid == question_id:
                    qrels[str(question_id)][str(pid)] = 1
        return qrels


class FVQALoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.fvqa import parse_pairs, get_collection
        pairs = parse_pairs(self.args.anno_file, self.args.image_dir)
        qid2pid = {}
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            if self.args.image_dir is not None:
                self.images[d["q_id"]] = d['image']
            qid2pid[d["q_id"]] = d["passage_ids"]
        self.qid2pid = qid2pid

        collection, pid2passage = get_collection(self.args.all_blocks_file)
        self.pid2passage = pid2passage
        self.passage_ids = list(pid2passage.keys())
        self.collection = Collection(data=list(pid2passage.values()))

    def create_qrels(self, qids, I, passage_ids):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {"placeholder": 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                if pid in self.qid2pid[question_id]:
                    qrels[str(question_id)][str(pid)] = 1
        return qrels


class REMUQLoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.vqa_ret import parse_pairs, get_collection, gen_qrels
        pairs = parse_pairs(self.args.anno_file)
        # Load image 
        
        with open("data/webqa/imgs.lineidx", "r") as fp_lineidx:
            lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        fp = open("data/webqa/imgs.tsv", "r")

        meta = {}
        for d in pairs:
            meta[d["q_id"]] = d['golden_pid']
            self.queries[d["q_id"]] = d["question"]
            fp.seek(lineidx[int(d["image"]) % 10000000])
            imgid, img_base64 = fp.readline().strip().split("\t")
            self.images[d["q_id"]] = BytesIO(base64.b64decode(img_base64))
        fp.close()

        import pandas as pd
        _pid2passage = pd.read_csv(self.args.all_blocks_file).to_dict()
        pid2passage = {}
        for kid, text in zip(_pid2passage['kid'].values(), _pid2passage['text'].values()):
            pid2passage[str(kid)] = text
        self.pid2passage = pid2passage
        self.passage_ids = list(pid2passage.keys())
        self.collection = Collection(data=list(pid2passage.values()))
        self.meta = meta

    def create_qrels(self, qids, I, passage_ids):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {"placeholder": 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                if pid == self.meta[question_id]:
                    qrels[str(question_id)][str(pid)] = 1
        return qrels


class FashionIQLoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.fashioniq import parse_pairs, get_collection
        collection, passage_ids, pid2passage = get_collection(self.args.all_blocks_file, img_dir=self.args.image_dir, return_ids=True)
        self.pid2passage = pid2passage
        self.passage_ids = passage_ids
        self.collection = collection
        pairs = parse_pairs(self.args.anno_file, img_dir=self.args.image_dir)
        meta = {}
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            self.images[d["q_id"]] = d["image"]
            meta[d["q_id"]] = d['golden_id']
        self.meta = meta

    def create_qrels(self, qids, I, passage_ids):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {"placeholder": 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                if pid == self.meta[question_id]:
                    qrels[str(question_id)][str(pid)] = 1
        return qrels
    
class OKVQAGSLoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.vqa_ret import parse_pairs, get_collection
        
        pairs = parse_pairs(self.args.anno_file)
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            if self.args.image_dir is not None:
                self.images[d["q_id"]] = f"{self.args.image_dir}/{d['image']}.jpg"

        collection, passage_ids, pid2passage = get_collection(self.args.all_blocks_file, return_ids=True)
        self.qid2answers = {int(s['q_id']): s['answers'] for s in pairs}
        self.collection = collection
        self.passage_ids = passage_ids
        self.pid2passage = pid2passage

    def create_qrels(self, qids, I, passage_ids):
        from dataset.vqa_ret import gen_qrels
        passage_ids = [str(k) for k in passage_ids]
        
        qrels, cases = gen_qrels(qids, I, passage_ids, qid2answers=self.qid2answers, pid2passage=self.pid2passage, return_cases=True)
        
        self.meta = cases
        return qrels
    
class InfoSeekLoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.infoseek import parse_pairs, get_collection
        from dataset.vqa_ret import gen_qrels

        collection, passage_ids, pid2passage = get_collection(self.args.all_blocks_file, return_ids=True)
        self.collection = collection
        self.pid2passage = pid2passage
        self.passage_ids = passage_ids

        pairs = parse_pairs(self.args.anno_file)

        not_exist_images = 0
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            if self.args.image_dir is not None:
                image_path = f"{self.args.image_dir}/{d['image']}"
                if not os.path.exists(image_path):
                    image_path = ".".join(image_path.split(".")[:-1]) + ".jpg"
                    if not os.path.exists(image_path):
                        not_exist_images += 1
                self.images[d["q_id"]] = image_path
        print("Not exists images:", not_exist_images, "/", len(pairs))
        self.qid2answers = {int(s['q_id']): s['answers'] for s in pairs}

    def create_qrels(self, qids, I, passage_ids):
        from dataset.vqa_ret import gen_qrels
        
        return gen_qrels(qids, I, passage_ids, qid2answers=self.qid2answers, pid2passage=self.pid2passage)


class Vid2RLoader(BaseDatasetEvalLoader):
    def load_data(self):
        from dataset.vid2r import parse_pairs
        pairs, pid2passage = parse_pairs(self.args.anno_file)
        meta = {}
        for d in pairs:
            self.queries[d["q_id"]] = d["question"]
            meta[d["q_id"]] = d["q_id"]

        self.meta = meta
        self.pid2passage = pid2passage
        self.passage_ids = list(pid2passage.keys())
        self.collection = Collection(data=list(pid2passage.values()))

    def create_qrels(self, qids, I, passage_ids):
        from collections import defaultdict
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {"placeholder": 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                if str(pid) == str(self.meta[question_id]):
                    qrels[str(question_id)][str(pid)] = 1
        return qrels
    
from datasets import load_dataset
class M2KRLoader(BaseDatasetEvalLoader):
    repo_id = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
    
    def parse_pairs(self, subset, image_dir, split="test", use_instruction=False):
        samples = load_dataset(self.repo_id, subset+"_data", split=split)
        
        data = []
        for sample in tqdm(samples, desc="Parsing pairs"):
            q_id = sample["question_id"]
            img_path = sample["img_path"]
            if subset in ["Infoseek"]:
                img_path = os.path.basename(img_path)
            img_path = os.path.join(image_dir, img_path)
            question = sample["question"]
            if use_instruction:
                question = sample["instruction"] + " " + question
            pos_passages = sample["pos_item_contents"]
            golden_pid = sample["pos_item_ids"]
            
            data.append({
                "question": question, "document": pos_passages[0],
                "image": img_path, "q_id": q_id, "golden_pid": golden_pid[0]
            })           
            
            if "answers" in sample:
                answers = sample["answers"]
                data[-1]["answers"] = answers
            
        return data
    
    def load_data(self):
        subset = self.args.dataset_name
        pairs = self.parse_pairs(subset, image_dir=self.args.image_dir, use_instruction=self.args.use_instruction)
        meta = {}
        for d in pairs:
            meta[d["q_id"]] = {'golden_pid': d['golden_pid']}
            if "answers" in d:
                meta[d["q_id"]] = {'answers': d["answers"]}
            self.queries[d["q_id"]] = d["question"]
            self.images[d["q_id"]] = d["image"]
            
        _pid2passage = load_dataset(self.repo_id, subset+"_passages", split="test_passages")
        self.pid2passage = {}
        for kid, text in zip(_pid2passage['passage_id'], _pid2passage['passage_content']):
            self.pid2passage[str(kid)] = text
        
        if "answers" in pairs[0]:
            self.qid2answers = {s['q_id']: s['answers'] for s in pairs}
        
        self.passage_ids = list(self.pid2passage.keys())
        self.collection = Collection(data=list(self.pid2passage.values()))
        self.meta = meta

    def create_qrels(self, qids, I, passage_ids):
        if self.args.dataset_name in ["EVQA", "Infoseek"]:
            # from dataset.vqa_ret import gen_qrels

            passage_ids = [str(k) for k in passage_ids]
            # return gen_qrels(qids, I, passage_ids, qid2answers=self.qid2answers, pid2passage=self.pid2passage)
            return self.compute_pseudo_recall(qids, I, passage_ids)
        
        from collections import defaultdict
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(qids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {"placeholder": 0}
            for retrieved_id in retrieved_ids:
                pid = passage_ids[retrieved_id]
                if str(pid) == str(self.meta[question_id]):
                    qrels[str(question_id)][str(pid)] = 1
        return qrels