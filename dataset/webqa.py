import json, random, os
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
from dataset.base import BaseRetrievalDataset


def check_and_save_image(image_id, tsv_fp, lineidx, save_dir, save_type):
    tsv_fp.seek(lineidx[int(image_id)%10000000])
    img_id, img_base64 = tsv_fp.readline().strip().split("\t")
    assert int(image_id)==int(img_id)
    
    if save_type=="bytes":
        with open(os.path.join(save_dir, str(image_id)), "w") as fout:
            fout.write(img_base64)
    elif save_type=="jpg":
        image = Image.open(BytesIO(base64.b64decode(img_base64)))
        image.save(os.path.join(save_dir, str(image_id)+".jpg"))
    elif save_type=="png":
        image = Image.open(BytesIO(base64.b64decode(img_base64)))
        image.save(os.path.join(save_dir, str(image_id)+".png"))
    else:
        raise NotImplementedError("save_type should be bytes or jpg.")
    

def prepare_image_data(data_path, img_lineidx_path, img_tsv_path, save_dir, save_type="bytes"):
    with open(img_lineidx_path, "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        
    with open(data_path, "r") as f:
        samples = json.load(f)
        
    os.makedirs(save_dir, exist_ok=True)

    with open(img_tsv_path, "r") as fp:
        for sample in tqdm(samples.values()):
            if 'img_posFacts' in sample:
                for im in sample['img_posFacts']:
                    check_and_save_image(int(im['image_id']), fp, lineidx, save_dir, save_type)
            
            for im in sample['img_negFacts']:
                check_and_save_image(int(im['image_id']), fp, lineidx, save_dir, save_type)
    

class WebQADataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer,
                 img_processor, split="train|val", img_type="bytes"):
        """
        img_type: jpg | bytes
        """
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer)
        self.img_type = img_type

        with open(data_path, "r") as f:
            samples = json.load(f)

        for _, sample in tqdm(samples.items(), desc="Processing samples"):
            if not sample["split"] in split:
                continue

            if self.has_both_posFacts(sample):
                raise RuntimeError("Both 'img_posFacts' and 'txt_posFacts' cannot be true simultaneously.")

            self.process_sample(sample)

    def has_both_posFacts(self, sample):
        return 'img_posFacts' in sample and sample["img_posFacts"] and 'txt_posFacts' in sample and sample["txt_posFacts"]

    def process_sample(self, sample):
        """
        T_q: textual query, A: answer, T_d: positive document, I_d: positive reference image,
        T_d_neg: negative documents, T_d_neg: negative reference images
        """
        T_q = sample["Q"].replace('"', "")
        A = sample["A"][0].replace('"', "")

        T_d, I_d = self.get_posFacts(sample)
        # T_d_neg = [fa["title"].strip()+"\n\n"+fa["fact"].strip() for fa in sample['txt_negFacts']]
        # I_d_neg = [self.format_image_path(int(im['image_id'])) for im in sample["img_negFacts"]]

        for i in range(len(T_d)):
            self.data.append({
                "T_q": T_q, "A": A, "I_d": I_d[i] if I_d else [],
                "T_d": T_d[i], "I_q": [],
                # "T_d_neg": T_d_neg, "I_d_neg": I_d_neg
            })

    def get_posFacts(self, sample):
        T_d, I_d = [], []
        if 'img_posFacts' in sample:
            for im in sample['img_posFacts']:
                I_d.append(self.format_image_path(int(im['image_id'])))
                T_d.append(im["caption"].strip())
        if 'txt_posFacts' in sample:
            for fa in sample["txt_posFacts"]:
                # fa["title"].strip()+"\n\n"+
                T_d.append(fa["fact"].strip())
                
        return T_d, I_d

    def format_image_path(self, image_id):
        image_id = str(image_id)
        image_file = image_id if self.img_type=="bytes" else f"{image_id}.{self.img_type}"
        return os.path.join(self.img_dir, image_file)