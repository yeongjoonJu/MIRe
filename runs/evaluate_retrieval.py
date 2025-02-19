import argparse
import json, os
from io import BytesIO
import base64
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher
from dataset.base import load_jsonl
import pytrec_eval
import numpy as np
from tqdm import tqdm

from dataset.eval_loaders import *

LOADER_MAP = {
    "okvqa": OKVQALoader,
    "aokvqa": AOKVQALoader,
    "fvqa": FVQALoader,
    "remuq": REMUQLoader,
    "okvqa_gs": OKVQAGSLoader,
    "fashioniq": FashionIQLoader,
    "vid2r": Vid2RLoader,
    "OVEN": M2KRLoader,
    "Infoseek": M2KRLoader,
    "EVQA": M2KRLoader
}


def search(queries, collection, args, images=None, image_retrieval=False):
    nbits = int(args.index_name.split("nbits=")[1])
    config = ColBERTConfig(nbits=nbits, doc_maxlen=384, query_maxlen=64)
    
    queries = {key: queries[key] for key in sorted(queries)}
    if args.image_dir is not None:
        images = {key: images[key] for key in sorted(images)}

    # Search
    print("Searching top-100...")
    with Run().context(RunConfig(experiment=args.index_name.split(".")[0])):
        if args.mire_ckpt is not None:
            if "sent" in args.mire_ckpt.lower():
                from retrievers.indexing_modern import XknowSearcher
            else:
                from retrievers.indexing import XknowSearcher

            searcher = XknowSearcher(index=args.index_name, checkpoint=args.mire_ckpt, \
                                    config=config, collection=collection, image_retrieval=image_retrieval)
            ranking = searcher.search_all(queries=queries, images=images, k=100,)
        else:
            searcher = Searcher(index=args.index_name, checkpoint=args.colbert_ckpt, config=config, collection=collection)
            ranking = searcher.search_all(queries=queries, k=100,)
    
    ranking_dict = ranking.flat_ranking
    data = {}
    for items in ranking_dict:
        qid = items[0]
        if qid in data:
            data[qid]["I"].append(items[1])
            data[qid]["D"].append(items[-1])
        else:
            data[qid] = {"I": [items[1]], "D": [items[-1]]}

    return data, ranking_dict

def evaluate_and_print_metrics(qrels, run, eval_metrics, ret_rank):
    if ret_rank == 5:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'P_5', 'recall_5'})
        metrics = evaluator.evaluate(run)
        eval_metrics['MRR'] = np.average([v['recip_rank'] for v in metrics.values()])
        eval_metrics["P@5"] = np.average([v['P_5'] for v in metrics.values()])
        eval_metrics["R@5"] = np.average([v['recall_5'] for v in metrics.values()])
    elif ret_rank == 1:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"P_1"})
        metrics = evaluator.evaluate(run)
        eval_metrics["P@1"] = np.average([v['P_1'] for v in metrics.values()])
    else:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f'recall_{ret_rank}'})
        metrics = evaluator.evaluate(run)
        eval_metrics[f"R@{ret_rank}"] = np.average([v[f'recall_{ret_rank}'] for v in metrics.values()])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--index_name", type=str, default="okvqa.nbits=2")
    parser.add_argument("--anno_file", type=str, default="data/okvqa/test2014_pairs_cap_combine_sum.txt")
    parser.add_argument("--all_blocks_file", type=str, default="data/okvqa/all_blocks.txt")
    parser.add_argument("--save_path", type=str, default="results/eval_okvqa_mire_clip.json")
    parser.add_argument("--image_dir", type=str, default=None, help="data/wiki/wikipedia_images_full")
    parser.add_argument("--mire_ckpt", type=str, default=None)
    parser.add_argument("--save_pairs", action='store_true')
    parser.add_argument("--use_instruction", action='store_true')
    parser.add_argument("--dataset_name", type=str, default="okvqa", help="okvqa|aokvqa|remuq|okvqa_gs|infoseek")
    args = parser.parse_args()
    
    # Dataset loader
    if args.dataset_name not in LOADER_MAP:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    loader = LOADER_MAP[args.dataset_name](args)
    loader.load_data()

    image_retrieval = False
    if args.dataset_name in ["fashioniq"]:
        image_retrieval = True

    data, ranking_dict = search(loader.queries, loader.collection, args, images=loader.images, image_retrieval=image_retrieval)

    eval_metrics = {}
    qids = list(data.keys())

    # Evaluate for each rank
    for ret_rank in [1, 5, 10, 20, 50, 100]:
        print("Evaluating Rank", ret_rank)
        I, D = [], []
        for qid in qids:
            I.append(data[qid]["I"][:ret_rank]) # I: retrieved passage ids
            D.append(data[qid]["D"][:ret_rank]) # D: retrieval scores
            
        qrels = loader.create_qrels(qids, I, loader.passage_ids)

        if args.save_pairs and ret_rank == 5 and hasattr(loader, 'pid2passage') and loader.pid2passage:
            top5_docs = []
            for q_i, r_i, s_i in zip(qids, I, D):
                item = {
                    "question_id": q_i,
                    "question": loader.queries[q_i],
                    "passages": [],
                    "pid": [],
                    "retrieval_score": []
                }
                if loader.meta is not None:
                    item["meta"] = loader.meta[q_i]
                for r, s in zip(r_i, s_i):
                    p_i = loader.passage_ids[r]
                    passage = loader.pid2passage.get(p_i, "")
                    item['pid'].append(p_i)
                    item['passages'].append(passage)
                    item["retrieval_score"].append(s)
                
                if "correct" in qrels[str(q_i)]:
                    item["correct"] = qrels[str(q_i)]["correct"]

                top5_docs.append(item)
            with open(args.save_path[:-5]+"_docs.json", "w") as fout:
                json.dump(top5_docs, fout, indent=2, ensure_ascii=False)

        run = {}
        for qid, retrieved_ids, scores in zip(qids, I, D):
            run[str(qid)] = {loader.passage_ids[retrieved_id]: float(score) for retrieved_id, score in zip(retrieved_ids, scores)}

        evaluate_and_print_metrics(qrels, run, eval_metrics, ret_rank)

    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

    with open(args.save_path, "w") as fout:
        json.dump(eval_metrics, fout, ensure_ascii=False, indent=2)