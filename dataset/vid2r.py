import json, os
import numpy as np
from tqdm import tqdm
import torch
import random
from dataset.base import BaseRetrievalDataset
# from retrievers.vilt.transforms.pixelbert import pixelbert_transform

def parse_pairs(filename, random_samples=5000):
    with open(filename, "r") as f:
        samples = json.load(f)
    
    data = []
    passages = {}
    for i, sample in enumerate(samples):
        question = sample['question']
        if len(question) < 25:
            continue
          
        q_id = i
        passage = sample["document"].strip()
        
        data.append({"question": question, "q_id": q_id, "golden_passage": passage})
        passages[str(q_id)] = passage
        
        if len(data) >= random_samples:
            break
    
    return data, passages
            

class ViD2RDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False, nways=1):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        self.nways = nways
        
        # Caching
        if os.path.exists(data_path[:-4]+"pt"):
            self.data = torch.load(data_path[:-4]+"pt")
        else:
            with open(data_path, "r") as f:
                samples = json.load(f)
                
            for sample in tqdm(samples, desc="Processing samples"):
                self.process_sample(sample)
            torch.save(self.data, data_path[:-4]+"pt")
        
            del samples
        

    def process_sample(self, sample):
        question = sample["question"]
        passage = sample["document"].strip()
            
        image_path = sample["image"]
            
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)

        T_q_ids = self.tensorize_query(question)
        T_d_ids = self.tensorize_doc(passage)
        
        tensorized_data = {
            "T_q_ids": T_q_ids, "T_d_ids": T_d_ids,
            "I_q": q_image_path, "I_d": [], # "A": answer, 
        }
            
        self.data.append(tensorized_data)
        
        
    def tensorize_doc_w_mask(self, text, response=None):
        max_len = self.doc_tokenizer.doc_maxlen
        doc = ". " + text

        # 토크나이즈 시 오프셋 매핑을 요청 (response가 있을 때만)
        encoding = self.doc_tokenizer.tok(
            [doc],
            padding="longest",
            truncation="longest_first",
            return_tensors="pt",
            max_length=max_len,
            return_offsets_mapping=(response is not None)
        )

        ids = encoding.input_ids  # shape: (1, seq_len)
        # [D] marker 적용
        ids[:, 1] = self.doc_tokenizer.D_marker_token_id

        # response가 없는 경우 마스크 없이 반환
        if response is None:
            return ids[0]
        
        if len(response) == 0:
            return None

        offsets = encoding.offset_mapping[0]  # shape: (seq_len, 2)
        seq_len = ids.size(1)
        mask = [0] * seq_len
        offs_list = offsets.tolist()

        # 문서 내 response가 여러 번 등장할 수 있으므로 반복 탐색
        search_start = 0
        x = 0
        while True:
            start_char = doc.find(response, search_start)
            if start_char == -1:  # 더 이상 매칭 없음
                break
            end_char = start_char + len(response)

            # 토큰 범위 찾기
            start_token_idx, end_token_idx = None, None
            for i, (s, e) in enumerate(offs_list):
                if start_token_idx is None and s <= start_char < e:
                    start_token_idx = i
                if start_token_idx is not None and s < end_char <= e:
                    end_token_idx = i
                    break

            # end_token_idx를 아직 못 찾은 경우, 해당 substring이 여러 토큰에 걸쳐 있을 가능성
            try:
                if start_token_idx is not None and end_token_idx is None:
                    for i2 in range(start_token_idx, len(offs_list)):
                        s, e = offs_list[i2]
                        if e >= end_char:
                            end_token_idx = i2
                            break
            except Exception as e:
                print(e)
                print(ids.shape)
                print(response)
                

            # 유효한 token 범위가 존재한다면 마스크 갱신
            if start_token_idx is not None and end_token_idx is not None:
                for idx in range(start_token_idx, end_token_idx + 1):
                    mask[idx] = 1

            # 다음 매칭을 찾기 위해 search_start를 현재 찾은 response 뒤로 이동
            search_start = end_char
            x += 1
            if x>10:
                return None

        return ids[0], torch.tensor(mask)