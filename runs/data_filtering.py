import argparse
import json, re
import random
from tqdm import tqdm
from dataset.base import load_jsonl
from retrievers.colbert.data import Queries, Collection
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher, Indexer
import spacy

alpha = ['a','b','c','d','e','f','g','h', 'i', 'j', 'k', "l"]

def is_bad_question(question):
    if "a short description for this region:" in question:
        return True
    
    prefixs = ["How many", "What color", "What is the color", "What are the colors"]
    for prefix in prefixs:
        if prefix in question:
            return True
    
    return False

def filter_1st_stage(data, args):
    miniset = []
    doc_list = []
    nlp = spacy.load("en_core_web_sm")
    
    for c, sample in tqdm(enumerate(data)):
        if not "image" in sample or "VG_100K" in sample["image"]:
            continue
        
        conv = sample["conversations"]
        if not "id" in sample:
            sample["id"] = c
        for t in range(0, len(conv)-1, 2):
            if conv[t]["from"]!="human" or conv[t+1]["from"]!="gpt":
                break
            
            doc = conv[t+1]["value"].strip().lower()
            question = conv[t]["value"].replace("<image>", "").strip()
            
            if "unanswerable." in doc or "answering does not require reading" in doc:
                continue
            
            # Remove unnecessary prefixes
            if doc[-1]==".":
                ref = doc[:-1]
            if ref.strip().isdigit():
                continue
                        
            doc = re.sub(r"(yes|no|answerable)([,.!])", "", doc)            
            doc = re.sub(r"(certainly|sure)[!.]", "", doc) # Absolutely
            doc = re.sub(r"in (the|this) (image|scene|photo|picture)([,\s])?", "", doc)
            doc = re.sub(r"the image (showcases|features|captures|depicts|shows|presents|displays) ", "", doc)
            
            if not doc.strip():
                continue
            
            if len(doc.split()) <= 4:
                if "\n" in question:
                    phrase = question.split("\n")[0]
                tokens = nlp(phrase)
                nouns = [token.text for token in tokens if token.pos_ in ["NOUN", "PROPN"]]
                if nouns:
                    doc = f"{doc} ({', '.join(nouns)})"
            
            if len(doc) < args.drop_len or is_bad_question(question):
                continue
            
            doc = doc.strip()
            doc = doc[0].upper() + doc[1:]
            
            if "," in question:
                doc = doc.replace(question.split(", ")[0], "")
                                    
            mini_sample = {
                "image": sample["image"],
                "id": f"{sample['id']}_{t}",
                "question": question,
                "document": doc,
            }
            miniset.append(mini_sample)
            doc_list.append(doc)
            
    return miniset, doc_list
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", nargs="+", type=str, required=True)
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--save_path", type=str, default="data/vid2r/vid2r_wo_conversion.json")
    parser.add_argument("--n_bits", type=int, default=2)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--n_cands", type=int, default=10)
    parser.add_argument("--drop_len", type=int, default=30)
    parser.add_argument("--save_negative", action="store_true")
    args = parser.parse_args()
    
    # Load data
    data = []
    for data_path in args.data_paths:
        ext = data_path.split(".")[-1]
        if ext=="json":
            with open(data_path, "r") as fin:
                data.extend(json.load(fin))
        elif ext=="jsonl":
            data.extend(load_jsonl(data_path))
    
    filtered, doc_list = filter_1st_stage(data, args)
    
    print(f"{len(data)} -> {len(filtered)}")
        
    # Align keys
    key2img = {}
    for sample in filtered:
        if sample["image"] in key2img:
            key2img[sample["image"]].append(sample)
        else:
            key2img[sample["image"]] = [sample]

    for k, sample in enumerate(filtered):
        ids = [n["id"].strip() for n in key2img[sample["image"]]]
        if len(set(ids))!=len(ids):
            filtered[k]["id"] = f"{sample['id']}_{random.choice(alpha)}{random.randint(0,9)}"
    
    keyimg_dict = {}
    for k, v in key2img.items():
        keyimg_dict[k] = [d.copy() for d in v]
    
    if args.save_negative:
        # Add negative samples
        for i in range(len(filtered)):
            negs = keyimg_dict[filtered[i]["image"]]
            filtered[i]["negs"] = [ n for n in negs if filtered[i]["question"]!=n["question"]]
        
    with open(args.save_path, "w") as fout:
        json.dump(filtered, fout, indent=2, ensure_ascii=False)