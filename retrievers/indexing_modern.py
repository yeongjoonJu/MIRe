import os, json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SequentialSampler
from transformers import CLIPImageProcessor, AutoConfig

from retrievers.colbert import ColBERTConfig, ModernColBERT
from retrievers.colbert.data import Queries
from retrievers.colbert.modeling.checkpoint import Checkpoint, _stack_3D_tensors
from retrievers.colbert.indexing.collection_indexer import CollectionIndexer
from retrievers.colbert.indexing.collection_encoder import CollectionEncoder
from retrievers.colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from retrievers.colbert.indexer import Indexer
from retrievers.colbert.searcher import Searcher, TextQueries
from retrievers.colbert.infra.launcher import Launcher
from retrievers.colbert.utils.amp import MixedPrecisionManager
from retrievers.mire import SentMIRe
from retrievers.colbert.infra.run import Run
from retrievers.colbert.utils.utils import batch
from dataset.base import ImageCollection

from retrievers.indexing import MultimodalCollectionEncoder
    

class ColBERTforDoc(ModernColBERT):
    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
        
    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)


class XknowCheckpoint(SentMIRe):
    def __init__(self, model_config, colbert_config=None, image_retrieval=False):
        super().__init__(model_config, colbert_config)
        
        # self.colbert_config in baseconfig
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_config.vision_model_name)

        self.amp_manager = MixedPrecisionManager(True)
        self.img_retrieval = image_retrieval

    def query(self, input_ids, attn_mask, images=None, to_cpu=False):
        with torch.no_grad():
            with self.amp_manager.context():
                Q_t, Q_i = super().embed_VL(input_ids, attn_mask, images)                   
                Q = super().query(Q_t, Q_i, attn_mask.to(Q_t.device), img_retrieval=self.img_retrieval)
                return Q.cpu() if to_cpu else Q

    def doc(self, input_ids=None, attention_mask=None, images=None, keep_dims=True, to_cpu=False):
        assert input_ids is not None or images is not None
        
        with torch.no_grad():
            with self.amp_manager.context():
                D_t, D_i = super().embed_VL(input_ids, attention_mask, images)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(D_t.device)
                D = super().doc(D_t, D_i, attention_mask, keep_dims=keep_dims)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D
            
    def load_images(self, paths, bsize=None):
        images = [Image.open(path).convert("RGB") for path in paths]
        _images = []
        for v in self.image_processor(images)['pixel_values']: # tqdm
            _images.append(torch.tensor(v).unsqueeze(0))
        # _img_attn_mask = [1. for _ in range(len(paths))]
        batches = []
        if bsize is not None:
            for b in range(0, len(_images), bsize):
                batches.append(torch.cat(_images[b:b+bsize], dim=0))
                # torch.tensor(_img_attn_mask[b:b+bsize]).unsqueeze(-1)
            return batches
        else:
            return torch.cat(_images, dim=0) #, torch.tensor(_img_attn_mask).unsqueeze(-1)

    def queryFromText(self, queries, context=None, bsize=None, to_cpu=False, full_length_search=False):
        """
        context -> image
        """
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=None, bsize=bsize, full_length_search=full_length_search)
            if context is not None:
                batch_imgs = self.load_images(context, bsize=bsize)
                batches = [self.query(input_ids.to(self.device), attention_mask.to(self.device), images=images.to(self.device), to_cpu=to_cpu)
                        for (input_ids, attention_mask), images in zip(batches, batch_imgs)]
            else:    
                batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
                
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=None, full_length_search=full_length_search)
        if context is not None:
            image = self.load_images([context] if type(context) is str else context)
        else:
            image = None
        
        return self.query(input_ids, attention_mask, images=image)
    
    def docFromImage(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims == 'flatten'
        batches = self.load_images(docs, bsize=bsize)
        
        returned_text = []
        if return_tokens:
            returned_text = docs.copy()
        
        keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
        batches = [self.doc(images=images.to(self.device), keep_dims=keep_dims_, to_cpu=to_cpu)
                    for images in tqdm(batches, disable=not showprogress)]
            
        D, mask = [], []
        for D_, mask_ in batches:
            D.append(D_)
            mask.append(mask_)

        D, mask = torch.cat(D), torch.cat(mask)

        doclens = mask.squeeze(-1).sum(-1).tolist()

        D = D.view(-1, self.colbert_config.dim)
        D = D[mask.bool().flatten()].cpu()

        return (D, doclens, *returned_text)
    

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
            
            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                    for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten':
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for b in batches for d in b]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()
            
class XknowCollectionIndexer(CollectionIndexer):
    def __init__(self, config: ColBERTConfig, collection, model_config, ckpt_path):
        super().__init__(config, collection, False)
        self.checkpoint = XknowCheckpoint(model_config, colbert_config=self.config)
        self.checkpoint.load_state_dict(torch.load(ckpt_path)['model'])
        self.checkpoint.eval()
        if self.use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        
        self.encoder = MultimodalCollectionEncoder(
            config, self.checkpoint, is_image=type(collection) is ImageCollection)
        
    def index(self):
        '''
        Encode embeddings for all passages in collection.
        Each embedding is converted to code (centroid id) and residual.
        Embeddings stored according to passage order in contiguous chunks of memory.

        Saved data files described below:
            {CHUNK#}.codes.pt:      centroid id for each embedding in chunk
            {CHUNK#}.residuals.pt:  16-bits residual for each embedding in chunk
            doclens.{CHUNK#}.pt:    number of embeddings within each passage in chunk
        '''
        with self.saver.thread():
            batches = self.collection.enumerate_batches(rank=self.rank)
            for chunk_idx, offset, passages in tqdm(batches, disable=self.rank > 0):
                if self.config.resume and self.saver.check_chunk_exists(chunk_idx):
                    Run().print_main(f"#> Found chunk {chunk_idx} in the index already, skipping encoding...")
                    continue
                # Encode passages into embeddings with the checkpoint model
                embs, doclens = self.encoder.encode_passages(passages) 
                if self.use_gpu:
                    assert embs.dtype == torch.float16
                else:
                    assert embs.dtype == torch.float32
                    embs = embs.half()

                Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                                 f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")

                self.saver.save_chunk(chunk_idx, offset, embs, doclens) # offset = first passage index in chunk
                del embs, doclens
        

def encode(config, collection, shared_lists, shared_queues, model_config, ckpt_path):
    encoder = XknowCollectionIndexer(
        config=config, collection=collection, model_config=model_config, ckpt_path=ckpt_path)
    encoder.run(shared_lists)
    
class XknowIndexer(Indexer):
    def __init__(self, checkpoint: str, config=None):
        super().__init__(os.path.dirname(checkpoint), config)
        
        if checkpoint[-3:]=='pth':
            dir_name = os.path.dirname(checkpoint)
        else:
            dir_name = checkpoint
        self.model_cfg = AutoConfig.from_pretrained(dir_name)
        self.ckpt_path = checkpoint
        
    def launch_process(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        # Encodes collection into index using the CollectionIndexer class
        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues, self.model_cfg, self.ckpt_path)
        
        
class XknowSearcher(Searcher):
    def __init__(self, index, checkpoint=None, collection=None, config=None, image_retrieval=False):
        super().__init__(index, os.path.dirname(checkpoint), collection=collection, config=config)
        
        dir_name = os.path.dirname(checkpoint)
        model_cfg = AutoConfig.from_pretrained(dir_name)
        model_cfg.pretraining = False
        self.checkpoint = XknowCheckpoint(model_cfg, colbert_config=self.config, image_retrieval=image_retrieval)
        self.checkpoint.load_state_dict(torch.load(checkpoint)['model'])
        self.checkpoint.eval()
        
        if self.config.total_visible_gpus > 0:
            self.checkpoint = self.checkpoint.cuda()
        
    def encode(self, text: TextQueries, img_queries=None, full_length_search=False):
        queries = text if type(text) is list else [text]
        if img_queries is not None:
            img_queries = [img_queries] if type(img_queries) is str else img_queries
            assert len(queries)==len(img_queries)
        
        bsize = 128 if len(queries) > 128 else None
                
        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, context=img_queries,\
                                        to_cpu=True, full_length_search=full_length_search)

        return Q
    
    def search(self, text:str, image_path:str, k=10, filter_fn=None, full_length_search=False, pids=None):
        Q = self.encode(text, image_path, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)
    
    def search_all(self, queries: TextQueries, images: dict, k=10, filter_fn=None, full_length_search=False, qid_to_pids=None):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        images = list(images.values())
        assert len(queries_)==len(images)
        
        Q = self.encode(queries_, img_queries=images, full_length_search=full_length_search)
        
        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids)
    
    
from dataset.base import DataPool
    
def generate_embeddings(model, query_tokenizer, doc_tokenizer, args, \
                per_gpu_eval_batch_size=256, write_to_file=False):
    
    dataset = DataPool(args.anno_file, args.all_blocks_file, \
        query_tokenizer, doc_tokenizer, passage_max_seq_length=args.doc_max_len)

    output_dir = f"experiments/{args.exp_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    eval_batch_size = per_gpu_eval_batch_size * torch.cuda.device_count()
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    # Eval!
    print("***** Gen passage rep *****")
    print("  Num examples = %d", len(dataset))
    print("  Batch size = %d", args.eval_batch_size)
    # run_dict = {}
    
    if write_to_file:
        fout = open(os.path.join(output_dir, "index.pt"), 'a')
        
    passage_ids = []
    p_embeds_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ids = np.asarray(batch['id']).reshape(-1).tolist()
        passage_ids.extend(ids)
        
        batch = {k: v.to(args.device) for k, v in batch.items() if k != 'id'}
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            p_embeds = model.doc(
                doc_ids=input_ids,
                doc_attention_mask=attention_mask
            )
            p_embeds = p_embeds.detach().cpu().tolist()
            p_embeds_list.extend(p_embeds)
        
        if write_to_file:
            for p_id, p_emb in zip(ids, p_embeds):
                fout.write(json.dumps({'id': p_id, 'rep': p_emb}) + '\n')
    
    if write_to_file:
        fout.close()
    
    return passage_ids, p_embeds_list

def rank_fusion(D, I, fusion):
    # D, I shape: (num_questions, num_objs, retrieve_top_k).
    print('Reshaped D.shape: {}'.format(
        ' '.join([str(d) for d in D.shape])))
    print('Reshaped I.shape: {}'.format(
        ' '.join([str(d) for d in I.shape])))
    
    num_questions, num_objs, k = D.shape
    
    fusion_scores = {}
    for qid in range(num_questions):
        for oid in range(num_objs):
            for pid in range(k):
                if qid not in fusion_scores:
                    fusion_scores[qid] = {}
                score = D[qid][oid][pid]
                retrieved_id = I[qid][oid][pid]
                
                if fusion == 'combine_max':
                    fusion_scores[qid][retrieved_id] = max(
                        fusion_scores[qid].get(retrieved_id, float('-inf')), score)
                elif fusion == 'combine_sum':
                    fusion_scores[qid][retrieved_id]= fusion_scores[qid].get(retrieved_id, 0.) + score
                else:
                    raise ValueError(
                        f'`fusion type` must be one of `combine_max` or `combine_sum`')
                                        
    fusion_D = []
    fusion_I = []
    for qid, scores in fusion_scores.items():
        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fusion_D.append([x[1] for x in ranked_scores[:k]])
        fusion_I.append([x[0] for x in ranked_scores][:k])

    return np.asarray(fusion_D), np.asarray(fusion_I)