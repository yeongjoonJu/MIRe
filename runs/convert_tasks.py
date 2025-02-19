import argparse
import json, re
import random
from tqdm import tqdm
from dataset.base import load_jsonl
from retrievers.colbert.data import Queries, Collection
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher, Indexer


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--db_pool", type=str, default="data/wiki/Wiki6M_ver_1_0.jsonl")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_bits", type=int, default=2)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--n_cands", type=int, default=3)
    parser.add_argument("--indexing", action="store_true")
    parser.add_argument("--remove_bracket", action="store_true")
    args = parser.parse_args()
    
    # Load data
    ext = args.data_path.split(".")[-1]
    if ext=="json":
        with open(args.data_path, "r") as fin:
            data = json.load(fin)
    elif ext=="jsonl":
        data = load_jsonl(args.data_path)
        
    wiki_db = load_jsonl(args.db_pool)
    wiki_db = [p['wikipedia_content'].strip() for p in wiki_db]

    collection = Collection(data=wiki_db)
    
    print(f"Loaded {len(collection):,} passages")
        
    # Indexing
    index_name = f"infoseek.nbits={args.n_bits}"
    config = ColBERTConfig(nbits=args.n_bits, doc_maxlen=256, query_maxlen=80)
    if args.indexing:
        with Run().context(RunConfig(nranks=args.n_ranks, experiment="infoseek")):
            indexer = Indexer(checkpoint=args.colbert_ckpt, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=False)
        print("Indexing is finished!")
    
    # Search
    with Run().context(RunConfig(experiment="infoseek")):
        searcher = Searcher(index=index_name, config=config, collection=collection)
        
    for i in tqdm(range(len(data))):
        results = searcher.search(data[i]["document"], k=args.n_cands)
        passages = [searcher.collection[p_id] for p_id in results[0]] # passage_ids
        passages = [".".join(p.split(".")[:2]) for p in passages]
        if args.remove_bracket:
            data[i]["document"] = re.sub(r"\([^(]+\)", "", data[i]["document"]).strip()
            unified = ["Title: "+data[i]["document"]+" content:"]
            unified.extend(passages)
        else:
            unified = [passages[0]]
            unified.append(data[i]['document'])
            unified.extend(passages[1:])
        
        data[i]['answer'] = data[i]['document']
        data[i]['document'] = " ".join(unified).replace("..", ".")
                
    with open(args.save_path, "w") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)