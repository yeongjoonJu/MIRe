import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from retrievers.colbert.modeling.colbert import colbert_score, colbert_score_reduce
from retrievers.colbert import ColBERT, ColBERTConfig, ModernColBERT
from retrievers.layers.vit import CLIPVisionTower
from retrievers.layers.siglip import SigLipVisionTower
from retrievers.colbert.utils.utils import flatten
from transformers import AutoConfig
from retrievers.colbert.infra import ColBERTConfig
import torch.distributed as dist
from typing import Optional
   

def build_vision_tower(model_name, **kwargs):
    # return CLIPVisionTower(model_name, **kwargs)
    if 'clip' in model_name:
        return CLIPVisionTower(model_name, **kwargs)
    else:
        return SigLipVisionTower(model_name, **kwargs)

    
def build_text_tower(text_tower_cfg: ColBERTConfig, **kwargs):
    return ColBERT(name=text_tower_cfg.checkpoint, colbert_config=text_tower_cfg)


class QueryGuidedAttentivePooling(nn.Module):
    def __init__(
        self, t_dim, v_dim, num_tokens=16, num_heads=8, out_dim=128
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.scale = math.sqrt(out_dim)
        self.num_heads = num_heads #8
        
        proj_dim = self.num_heads*out_dim * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(v_dim),
            nn.Linear(v_dim, proj_dim),
            nn.GELU()
        )
                
        self.k_state = nn.Linear(proj_dim, self.num_heads*out_dim, bias=False)
        self.v_state = nn.Linear(proj_dim, self.num_heads*out_dim, bias=False)
        self.o_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )
        
        self.split_mlp = nn.Sequential(
            nn.Linear(v_dim, (num_tokens*out_dim)//2),
            nn.Tanh(),
            nn.Linear((num_tokens*out_dim)//2, num_tokens*out_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def encode_image(self, v):
        # Split CLS
        v_cls = v[:,0:1]
        tokens = self.split_mlp(v_cls)
        return tokens.view(v.size(0), self.num_tokens, self.out_dim)
    
    
    def encode_mm(self, t, v, t_attn_mask, return_relevance=False):
        # Add latents
        v_feats = v[:,1:] # B x S x D
        B, S, D = v_feats.shape
        v_feats = self.proj(v_feats)
        
        query_layer = t.unsqueeze(2).repeat(1,1,self.num_heads,1)
        key_layer = self.k_state(v_feats).view(B, S, self.num_heads, self.out_dim) # B S head D
        value_layer = self.v_state(v_feats).view(B, S, self.num_heads, self.out_dim) # B S head D

        query_layer = query_layer.transpose(1,2)
        key_layer = key_layer.transpose(1,2)
        value_layer = value_layer.transpose(1,2)
        scores = torch.matmul(query_layer, key_layer.transpose(-2,-1)) / self.scale
        t_attn_mask = t_attn_mask.unsqueeze(1).unsqueeze(-1).float()
        scores = scores.masked_fill(t_attn_mask==0, -9999.0)
        
        attn_weight = F.softmax(scores, dim=-1) # B, H, Q, K
        attn_output = torch.matmul(attn_weight, value_layer) # B H Q K | B H K D -> B H Q D
        attn_output = attn_output + query_layer
        attn_output = attn_output.sum(dim=-2) / (t_attn_mask.sum(dim=-2)+1e-15)
        attn_output = self.o_proj(attn_output) + attn_output
        
        if return_relevance:
            return attn_output, attn_weight

        return (attn_output,)
    

    def forward(self, t, v, t_attn_mask, return_relevance=False):
        V_g = self.encode_image(v) # V_g
        mm_outputs = self.encode_mm(t, v, t_attn_mask, return_relevance)
        V_m = mm_outputs[0]
        
        
        Q =  torch.cat([V_g, V_m], dim=1)
        
        if return_relevance:
            return Q, mm_outputs[1]
        
        return Q
    
    
class MIRe(ColBERT):
    def __init__(self, config, colbert_config=None, negatives_cross_device=False):
        super().__init__(name=config.colbert_checkpoint, colbert_config=colbert_config)
        
        vision_config = AutoConfig.from_pretrained(config.vision_model_name).vision_config
        vis_hidden_size = vision_config.hidden_size
        hidden_size = config.hidden_size
        self.config = config
        self.pretraining = self.config.pretraining
        self.hidden_size = hidden_size
            
        self.vision_tower = build_vision_tower(config.vision_model_name)

        self.query_fusion = QueryGuidedAttentivePooling(
            t_dim=hidden_size, v_dim=vis_hidden_size,
            num_tokens=config.num_tokens,
            num_heads=config.num_heads
        )
        
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
    
    def freeze_parameters(self, frozen=True, linear_frozen=True):
        for _, param in self.vision_tower.named_parameters():
            param.requires_grad = False #self.pretraining
        
        for _, param in self.model.named_parameters():
            param.requires_grad = not frozen
        
        self.linear.requires_grad = not linear_frozen
        
    
    def freeze_vision_proj(self):
        self.query_fusion.split_mlp.requires_grad = False

    def tensorize_query(self, text):
        
        ids = self.raw_tokenizer(['. ' + text], padding="longest", truncation=False,
                                   return_tensors="pt").input_ids
        
        # postprocess for the [Q] marker and the [MASK] augmentation
        Q_marker_token_id = self.raw_tokenizer.convert_tokens_to_ids([self.colbert_config.query_token_id])[0]
        ids[:, 1] = Q_marker_token_id
        ids[ids == self.raw_tokenizer.pad_token_id] = self.raw_tokenizer.mask_token_id
        
        return ids

    def encode_query(self, input_ids=None, attention_mask=None, images=None, image_embeds=None):
        Q_t, Q_i = self.embed_VL(input_ids, attention_mask, images, image_embeds)
        return self.query(Q_t, Q_i, attention_mask)
    
    def encode_document(self, input_ids=None, attention_mask=None, images=None, image_embeds=None, image_attn_mask=None):
        D_t, D_i = self.embed_VL(input_ids, attention_mask, images, image_embeds)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device)
        return self.doc(D_t, D_i, mask, image_attn_mask, keep_dims='return_mask')
            
    def forward(
        self,
        query_ids=None,
        query_attention_mask=None,
        query_images=None,
        query_image_mask=None,
        doc_ids=None,
        doc_attention_mask=None,
        doc_images=None,
        doc_image_mask=None,
        query_image_embs=None,
        doc_image_embs=None,
        return_loss=False,
        return_ib_acc=False,
    ):
        # query
        Q = self.encode_query(query_ids, query_attention_mask, query_images, query_image_embs)
        # document
        D, D_mask = self.encode_document(doc_ids, doc_attention_mask, doc_images, doc_image_embs, doc_image_mask)
        
        if self.training and self.negatives_cross_device:
            Q = self._dist_gather_tensor(Q)
            D = self._dist_gather_tensor(D)
            D_mask = self._dist_gather_tensor(D_mask)
            
        nway = D.size(0) // Q.size(0)
        # Repeat each query encoding for every corresponding document.
        if Q.size(0)!=D.size(0):
            Q_duplicated = Q.repeat_interleave(nway, dim=0).contiguous()
        else:
            Q_duplicated = Q
        
        temperature = 1.0
        if self.pretraining:
            temperature = 0.3
        elif self.model.training:
            temperature = 0.8

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask, temperature=temperature)

        scores = self.score(Q_duplicated, D, D_mask)
        
        scores = scores / temperature
        scores = scores.view(-1, nway)
                
        loss = None
        accuracy = None
        if return_loss:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, labels)

            if self.colbert_config.use_ib_negatives:
                loss += ib_loss
            
            if return_ib_acc:
                accuracy = self.inbatch_accuracy(Q, D, D_mask)
            else:
                accuracy = torch.tensor(1.0)
        
        outputs = {'loss': loss, 'score': scores, "accuracy": accuracy}

        return outputs
    
    def embed_VL(self, input_ids, attention_mask, images=None, image_embeds=None):
        assert input_ids is not None or images is not None
        T, I = None, None
        
        if images is not None:
            # final [CLS]; outputs in the penultimate layer
            with torch.no_grad():
                I = self.vision_tower(images.to(self.device))
        
        if image_embeds is not None:
            I = image_embeds

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)            
            T = self.bert(input_ids, attention_mask=attention_mask)[0]

        return T, I
    
    def query(self, Q_t, Q_i, attention_mask, image_attn_mask=None):
        if Q_i is not None:                
            Q_t = self.linear(Q_t)
            if Q_t.size(1)!=attention_mask.size(1):
                B, L = attention_mask.size()
                one_pad = torch.ones((B, self.num_vis_tokens)).to(attention_mask.device)
                attention_mask = torch.cat([one_pad, attention_mask], dim=1)
                
            Q_t = Q_t * attention_mask.unsqueeze(2).float()
            Q_i = self.query_fusion(Q_t, Q_i, t_attn_mask=attention_mask)
            if self.pretraining:
                Q = Q_i
            else:
                Q = torch.cat([Q_i, Q_t], dim=1)
        else:
            Q = self.linear(Q_t)
            Q = Q * attention_mask.unsqueeze(2).float()
            
        return torch.nn.functional.normalize(Q, p=2, dim=2)
    
    def doc(self, D_t=None, D_i=None, mask=None, image_attn_mask=None, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']
        
        D = []
        
        if D_i is not None:
            D_vg = self.query_fusion.encode_image(D_i)
            input_ids = self.tensorize_query("Can you describe the primary elements in this image?")
            input_ids = input_ids.repeat(D_i.size(0), 1).to(D_i.device)
            attention_mask = torch.ones_like(input_ids)
            t = self.linear(self.bert(input_ids, attention_mask=attention_mask)[0])
            D_vm = self.query_fusion.encode_mm(t, D_i, attention_mask)[0]
            D_i = torch.cat([D_vg, D_vm], dim=1)
        
        if D_t is not None:
            D_t = self.linear(D_t)
            D_t = D_t * mask.unsqueeze(2).float()
            
        if D_i is not None and D_t is not None:
            D = torch.cat([D_i, D_t], dim=1)
            image_attn_mask = torch.ones((D_i.size(0), D_i.size(1)),
                                  device=mask.device, dtype=mask.dtype)
            mask = torch.cat([image_attn_mask, mask], dim=1)
        elif D_i is not None:
            D = D_i
            mask = torch.ones((D_i.size(0), D_i.size(1)), device=D.device)
        elif D_t is not None:
            D = D_t
                        
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
    
    def inbatch_accuracy(self, Q, D_padded, D_mask):
        B, S, E = Q.shape
        B, K, E = D_padded.shape
        Q = Q.view(-1, E)
        D_padded = D_padded.view(-1, E)
        
        scores = D_padded @ Q.t() # (B K) x (B S)
        scores = scores.view(B,K,B,S).permute(0,2,1,3) # B x B x K x S
        
        # D_mask -> [B x K]
        D_mask = D_mask.unsqueeze(1).expand(-1,B,-1)
        D_padding = ~D_mask.view(scores.size(0), scores.size(1), scores.size(2)).bool()
        scores[D_padding] = -9999
        
        scores = scores.max(2).values.sum(-1)
        _, max_idx = torch.max(scores, 1)
        labels = torch.arange(len(scores), device=scores.device)
        accuracy = (max_idx==labels.detach()).sum() / scores.size(0)
        
        return accuracy
    
    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    
class SentMIRe(ModernColBERT):
    def __init__(self, config, colbert_config=None, negatives_cross_device=False):
        super().__init__(name=config.colbert_checkpoint, colbert_config=colbert_config)
        
        vision_config = AutoConfig.from_pretrained(config.vision_model_name).vision_config
        vis_hidden_size = vision_config.hidden_size
        hidden_size = config.hidden_size
        self.config = config
        self.pretraining = self.config.pretraining
        self.hidden_size = hidden_size
            
        self.vision_tower = build_vision_tower(config.vision_model_name)

        self.query_fusion = QueryGuidedAttentivePooling(
            t_dim=hidden_size, v_dim=vis_hidden_size,
            num_tokens=config.num_tokens,
            num_heads=config.num_heads
        )
        
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
    
    def freeze_parameters(self, frozen=True, linear_frozen=True):
        for _, param in self.vision_tower.named_parameters():
            param.requires_grad = False #self.pretraining
        
        for _, param in self.model.named_parameters():
            param.requires_grad = not frozen
        
        self.dense.requires_grad = not linear_frozen
        
    
    def freeze_vision_proj(self):
        self.query_fusion.split_mlp.requires_grad = False
            
    
    def forward(
        self,
        query_ids=None,
        query_attention_mask=None,
        query_images=None,
        query_image_mask=None,
        doc_ids=None,
        doc_attention_mask=None,
        doc_images=None,
        doc_image_mask=None,
        query_image_embs=None,
        doc_image_embs=None,
        return_loss=False,
        return_ib_acc=False,
    ):
        # query
        Q_t, Q_i = self.embed_VL(query_ids, query_attention_mask,
                                 query_images, query_image_embs)
        
        # document
        D_t, D_i = self.embed_VL(doc_ids, doc_attention_mask, doc_images)
        if doc_image_embs is not None:
            D_i = doc_image_embs
        
        mask = torch.tensor(self.mask(doc_ids, skiplist=self.skiplist), device=self.device)
        D, D_mask = self.doc(D_t, D_i, mask, doc_image_mask, keep_dims='return_mask')
        
        Q = self.query(Q_t, Q_i, query_attention_mask)
        
        if self.training and self.negatives_cross_device:
            Q = self._dist_gather_tensor(Q)
            D = self._dist_gather_tensor(D)
            D_mask = self._dist_gather_tensor(D_mask)
            
        nway = D.size(0) // Q.size(0)
        # Repeat each query encoding for every corresponding document.
        if Q.size(0)!=D.size(0):
            Q_duplicated = Q.repeat_interleave(nway, dim=0).contiguous()
        else:
            Q_duplicated = Q
        
        temperature = 1.0
        if self.pretraining:
            temperature = 0.3
        elif self.model.training:
            temperature = 0.8

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask, temperature=temperature)
            
        scores = self.score(Q_duplicated, D, D_mask)
        
        scores = scores / temperature
        scores = scores.view(-1, nway)
                
        loss = None
        accuracy = None
        if return_loss:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, labels)
            
            if self.colbert_config.use_ib_negatives:
                loss += ib_loss
            
            if return_ib_acc:
                accuracy = self.inbatch_accuracy(Q, D, D_mask)
            else:
                accuracy = torch.tensor(1.0)
        
        outputs = {'loss': loss, 'score': scores, "accuracy": accuracy}

        return outputs
    
    def embed_VL(self, input_ids, attention_mask, images=None, image_embeds=None):
        assert input_ids is not None or images is not None
        T, I = None, None
        
        if images is not None:
            # final [CLS]; outputs in the penultimate layer
            with torch.no_grad():
                I = self.vision_tower(images.to(self.device))
        
        if image_embeds is not None:
            I = image_embeds

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            T = self.bert(input_ids, attention_mask=attention_mask)[0]

        return T, I
    
    def query(self, Q_t, Q_i, attention_mask, image_attn_mask=None, img_retrieval=False):
        if Q_i is not None:                
            Q_t = self.dense(Q_t).float()
            if Q_t.size(1)!=attention_mask.size(1):
                B, L = attention_mask.size()
                one_pad = torch.ones((B, self.num_vis_tokens)).to(attention_mask.device)
                attention_mask = torch.cat([one_pad, attention_mask], dim=1)
                
            Q_t = Q_t * attention_mask.unsqueeze(2).float()
            Q_i = self.query_fusion(Q_t, Q_i, t_attn_mask=attention_mask)
            if self.pretraining:
                Q = Q_i
            else:
                Q = torch.cat([Q_i, Q_t], dim=1)
        else:
            Q = self.dense(Q_t).float()
            Q = Q * attention_mask.unsqueeze(2).float()
            
        return torch.nn.functional.normalize(Q, p=2, dim=2)
    
    def doc(self, D_t=None, D_i=None, mask=None, image_attn_mask=None, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']
        
        D = []
        
        if D_i is not None:
            D_i = self.query_fusion.encode_image(D_i)
        
        if D_t is not None:
            D_t = self.dense(D_t).float()
            D_t = D_t * mask.unsqueeze(2).float()
        
        if D_i is not None and D_t is not None:
            D = torch.cat([D_i, D_t], dim=1)
            image_attn_mask = torch.ones((D_i.size(0), D_i.size(1)),
                                  device=mask.device, dtype=mask.dtype)
            mask = torch.cat([image_attn_mask, mask], dim=1)
        elif D_i is not None:
            D = D_i
            mask = torch.ones((D_i.size(0), D_i.size(1)), device=D.device)
        elif D_t is not None:
            D = D_t
                        
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
    
    def inbatch_accuracy(self, Q, D_padded, D_mask):
        B, S, E = Q.shape
        B, K, E = D_padded.shape
        Q = Q.view(-1, E)
        D_padded = D_padded.view(-1, E)
        
        scores = D_padded @ Q.t() # (B K) x (B S)
        scores = scores.view(B,K,B,S).permute(0,2,1,3) # B x B x K x S
        
        # D_mask -> [B x K]
        D_mask = D_mask.unsqueeze(1).expand(-1,B,-1)
        D_padding = ~D_mask.view(scores.size(0), scores.size(1), scores.size(2)).bool()
        scores[D_padding] = -9999
        
        scores = scores.max(2).values.sum(-1)
        _, max_idx = torch.max(scores, 1)
        labels = torch.arange(len(scores), device=scores.device)
        accuracy = (max_idx==labels.detach()).sum() / scores.size(0)
        
        return accuracy
    
    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors