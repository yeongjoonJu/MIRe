import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# from FlagEmbedding import FlagModel
# from flag_clip import Flag_clip
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
from retrievers.vista import Visualized_BGE
from dataset.eval_loaders import *
from runs.evaluate_retrieval import evaluate_and_print_metrics


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


logger = logging.getLogger(__name__)

@dataclass
class Args:
    resume_path: str = field(
        default="ckpts/vista/Visualized_base_en_v1.5.pth", 
        metadata={'help': 'The model checkpoint path.'}
    )
    image_dir: str = field(
        default="your_image_directory_path",
        metadata={'help': 'Where the images located on.'}
    )
    encoder: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=512, 
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )
    dataset_name: str = field(
        default="okvqa_gs",
        metadata={'help': 'Which dataset loader to use from LOADER_MAP.'}
    )
    # anno_file, all_blocks_file 등 기존 로더가 필요로 하는 인자들도 추가로 필요
    anno_file: str = field(
        default="data/okvqa/test2014_pairs_cap_combine_sum.txt",
        metadata={'help': 'Annotation file path for the dataset.'}
    )
    all_blocks_file: str = field(
        default="data/okvqa/all_blocks.txt",
        metadata={'help': 'Collection blocks path for the dataset.'}
    )
    use_instruction: bool = field(
        default=False,
        metadata={'help': 'Whether to use instruction or not.'}
    )

class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        tokens = self.tokenizer(
            sentence,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokens

def collate_fn(batch):
    input_ids = [item['input_ids'].squeeze(0) for item in batch]
    attention_mask = [item['attention_mask'].squeeze(0) for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded
    }

def index(model, corpus, batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode(text="test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        dataset = TextDataset(corpus, model.tokenizer, max_length)
        dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # for start_index in tqdm(range(0, len(corpus), batch_size), desc="Encoding", disable=len(corpus)<256):
        #     sentences_batch = corpus[start_index:start_index + batch_size]
        #     inputs = model.tokenizer(
        #         sentences_batch,
        #         padding=True,
        #         truncation=True,
        #         return_tensors='pt',
        #         max_length=max_length,
        #     )
        #     encoded_batches.append(inputs)

        text_corpus_embeddings = []
        for batch in tqdm(dataloader, desc="Inference Embeddings", disable=len(corpus)<256):
            inputs_padded = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = model.encode_text(inputs_padded)
            embeddings = embeddings.detach().cpu().numpy()
            text_corpus_embeddings.append(embeddings)
            
        # mm_it_corpus_embeddings = model.encode_corpus(mm_it_corpus, batch_size=batch_size, max_length=max_length, corpus_type='mm_it')
        corpus_embeddings = np.concatenate(text_corpus_embeddings, axis=0)
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index_all = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = False
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index_all = faiss.index_cpu_to_all_gpus(faiss_index_all, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index_all.train(corpus_embeddings)
    faiss_index_all.add(corpus_embeddings)

    return faiss_index_all


def search(model, queries, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    total_queries = len(queries)
    all_scores = []
    all_indices = []
    
    for start in tqdm(range(0, total_queries, batch_size), desc="Searching"):
        batch = queries[start: start + batch_size]
        images = []
        texts = []
        for item in batch:
            text_str = item["q_text"]
            texts.append(text_str)
            
            if "q_img" in item and item["q_img"] is not None:
                images.append(item["q_img"])
        
        if not images:
            images = None
            raise NotImplementedError

        image_tensors = []
        for img_path in images:
            img_data = Image.open(img_path).convert("RGB")
            img_tensor = model.preprocess_val(img_data).unsqueeze(0)
            image_tensors.append(img_tensor)
            
        images_tensor = torch.cat(image_tensors, dim=0).to(model.device)
        
        inputs = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(model.device)
        
        with torch.no_grad():
            query_embeddings = model.encode_mm(
                images=images_tensor,
                texts=inputs
            )
        query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)
        
        score, indice = faiss_index.search(query_embeddings, k=k)
        
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
    
def evaluate(preds, labels, cutoffs=[1,5,10,20,50,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    easy_recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list):
            label = [label]
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
            if len(recall) > 0:
                easy_recalls[k] += 1
    recalls /= len(preds)
    easy_recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall
    
    for i, cutoff in enumerate(cutoffs):
        easy_recall = easy_recalls[i]
        metrics[f"Easy_Recall@{cutoff}"] = easy_recall

    return metrics


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    
    if args.dataset_name not in LOADER_MAP:
        raise ValueError(f"Unknown dataset_name: {args.dataset_name}")
    loader_cls = LOADER_MAP[args.dataset_name]
    loader = loader_cls(args)
    loader.load_data()

    doc_texts = loader.collection.data
    doc_ids   = loader.passage_ids

    qids = sorted(loader.queries.keys())
    query_list = []
    for qid in qids:
        q_text = loader.queries[qid]
        q_img  = loader.images[qid] if (loader.images and qid in loader.images) else None

        query_list.append({
            "q_id": qid,
            "q_text": q_text,
            "q_img": q_img
        })
    
    model = Visualized_BGE(model_name_bge = args.encoder,
                        model_weight= args.resume_path,
                        normlized = True,)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(model.device)
    
    print(args.resume_path)
    
    faiss_index_all = index(
        model=model, 
        corpus=doc_texts, 
        batch_size=args.batch_size*8,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )

    all_scores, all_indices = search(
        model=model,
        queries=query_list,
        faiss_index=faiss_index_all,
        k=100,
        batch_size=args.batch_size,
        max_length=args.max_query_length
    )

    eval_metrics = {}
    for ret_rank in [1, 5, 10, 20, 50, 100]:
        print(f"Evaluating Rank = {ret_rank}")

        all_I = []
        all_D = []

        for i in range(len(query_list)):
            doc_indices = all_indices[i, :ret_rank]
            doc_scores = all_scores[i, :ret_rank]

            retrieved_ids = []
            retrieved_scores = []
            for idx, sc in zip(doc_indices, doc_scores):
                if idx < 0:
                    continue
                retrieved_ids.append(idx)       
                retrieved_scores.append(float(sc))
                
            all_I.append(retrieved_ids)
            all_D.append(retrieved_scores)

        qrels = loader.create_qrels(qids, all_I, doc_ids)

        # pytrec_eval run
        run = {}
        for qid, ret_ids, scores in zip(qids, all_I, all_D):
            run[str(qid)] = {}
            for doc_idx, sc in zip(ret_ids, scores):
                real_doc_id = doc_ids[doc_idx]
                run[str(qid)][str(real_doc_id)] = sc

        evaluate_and_print_metrics(qrels, run, eval_metrics, ret_rank)
        
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()