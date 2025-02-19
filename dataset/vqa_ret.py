import json, re
from collections import defaultdict
from retrievers.colbert.data import Collection
from dataset.infoseek import kb_id2img_path
import pandas as pd
from tqdm import tqdm
import random

def parse_pairs(filename, random_choice=False, caption=False):
    with open(filename, "r") as f:
        samples = json.load(f)
    
    data = []
    for sample in samples:
        q_id = sample["question_id"]
        img_id = sample["img_id"]
        question = sample["question"]
        if not caption and "caption" in sample:
            question = question.replace(sample["caption"], "").strip()
            
        passages = sample["ctxs"]
        if not passages:
            continue
        
        pos_passage = random.choice(passages) if random_choice else passages[0]
        pos_passage = pos_passage['text']
        # image_path = kb_id2img_path(f"Q{img_id}")
        golden_pid = sample["ctxs"][0]["id"]
        answers = list(sample["answers"].keys())

        data.append({
            "question": question, "document": pos_passage, "answers": answers,
            "image": img_id, "q_id": q_id, "golden_pid": golden_pid
        })

        if "caption" in sample:
            data[-1]["caption"] = sample["caption"]

        if "negs" in sample:
            data[-1]["negs"] = sample["negs"]
    

    return data

def get_collection(filename, return_ids=False):
    df = pd.read_csv(filename)
    passage_ids = df["kid"].tolist()
    passages = df["text"].tolist()
    
    print(f"Loaded {len(passage_ids)} passages.")
    
    collection = Collection(data=passages)
    
    if return_ids:
        passage_ids = [str(k) for k in passage_ids]
        id2passage = {k: v for k, v in zip(passage_ids, passages)}
        return collection, passage_ids, id2passage

    return collection


def has_answers(answers, passage):
    passage = passage.lower()
    for answer in answers:
        answer = answer.strip().lower()
        try:
            if re.search(r'\b{}\b'.format(answer), passage):
                return True
        except Exception as e:
            if answer in passage.lower():
                return True
    return False

def gen_qrels(question_ids, I, retrieved_id_to_passage_id, pid2passage=None, qid2answers=None, return_cases=False):
    qrels = defaultdict(dict)
    cases = {}
    for question_id, retrieved_ids in tqdm(zip(question_ids, I), total=len(question_ids)):
        if question_id not in qrels:
            qrels[str(question_id)] = {'placeholder': 0}

        is_correct = False
        for retrieved_id in retrieved_ids:
            passage_id = retrieved_id_to_passage_id[retrieved_id]
            # answers = qid2answers[int(question_id)]
            answers = qid2answers[question_id]
            passage = pid2passage[passage_id]

            if has_answers(answers, passage):
                qrels[str(question_id)][passage_id] = 1
                is_correct = True

        cases[question_id] = is_correct
    
    if return_cases:
        return qrels, cases
    
    return qrels