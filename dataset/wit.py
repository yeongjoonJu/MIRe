import argparse
import re, json
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from urllib import request
from PIL import Image
from io import BytesIO
import cairosvg
import os
from dataset.base import load_jsonl, convert_list2dict, BaseRetrievalDataset
from dataset.infoseek import kb_id2img_path
from datasets.utils.file_utils import get_datasets_user_agent
from datasets import Dataset
import requests, PIL
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def download_and_convert_to_jpg(url, save_path):
    # 이미지 다운로드
    try:
        response = request.urlopen(url).read()
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

    # BytesIO를 사용하여 이미지를 메모리에 로드
    try:
        image = Image.open(BytesIO(response))
    except OSError:
        # SVG 파일 처리
        try:
            # SVG를 PNG로 변환
            converted = cairosvg.svg2png(bytestring=response)
            image = Image.open(BytesIO(converted))
        except Exception as e:
            print(f"Error converting SVG to PNG: {e}")
            return False
    except Exception as e:
        print(e)
        return False

    # 이미지의 확장자 확인 및 JPG로 변환 (필요한 경우)
    if image.format != 'JPEG':
        image = image.convert('RGB')  # JPG 형식으로 변환
        save_path = os.path.splitext(save_path)[0] + '.jpg'

    # 이미지 저장
    try:
        image.save(save_path)
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

    return True
    

def inverse_cloze_task(data, title2img):
    # Only English
    data = data[data["language"] == "en"]

    titles = data["page_title"].tolist()
    contexts = data["context_page_description"].tolist()
    captions = data["caption_reference_description"].tolist()
    # img_urls = data["image_url"].tolist()

    converted = []
    for title, context, caption in tqdm(zip(titles, contexts, captions), total=len(titles)):
        if title in title2img:
            image_id = title2img[title]["wikidata_id"]
        else:
            continue
        
        # title이 context 안에 있는지 확인
        match = False
        if title in context:
            match = True

        # Match가 없는 경우, 처리
        if not match:
            # title에서 괄호 제거 후 재검색
            title = re.sub(r"\([\w\s]+\)", "", title).strip()
            if title in context:
                match = True

            if not match:
                # caption을 새로운 title로 사용
                title = caption.strip()
                if title in context:
                    match = True

                if not match: #  or not match.group()
                    continue
                
        context = re.sub(re.escape(title), "_rep_", context)
        splited = re.split(r"[.]\s", context)
            
        if len(splited) <= 1:
            continue
        
        for k, sent in enumerate(splited):
            if "_rep_" in sent:
                break
        
        question = splited[k].strip() + "."
        question = question.replace("_rep_", "_")
        document = ". ".join(splited[k+1:])
        document = document.strip()
        document = document.replace("_rep_", title)
        
        if document:
            converted.append({
                "question": question,
                "document": document,
                "image_id": image_id})

    missing_count = len(data) - len(converted)
    print(f"Total items missing: {missing_count}")

    return converted


class VL_ICT(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False,):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        with open(data_path, "r") as f:
            samples = json.load(f)
        
        # Caching
        if os.path.exists(data_path[:-4]+"pt"):
            self.data = torch.load(data_path[:-4]+"pt")
        else:
            for sample in tqdm(samples, desc="Processing samples"):
                self.process_sample(sample)
            torch.save(self.data, data_path[:-4]+"pt")
            
    def process_sample(self, sample):
        image_id = sample["image_id"]
        question = sample["question"]
        passage = sample["document"].strip()
        image_path = os.path.join(self.img_dir, kb_id2img_path(image_id))
        if os.path.exists(image_path):
            self.data.append({
                "T_q_ids": self.tensorize_query(question), "T_d_ids": self.tensorize_doc(passage),
                "I_q": image_path, "I_d": [], # "A": answer,
            })

def get_dirname_from_img_id(img_id):
    id = img_id[1:]
    if len(id) <= 2:
        return f"Q{id}"
    else:
        return f"Q{id[:3]}"

def list_of_dicts_to_dict_of_lists(list_of_dicts, default_value=None):
    all_keys = set(key for d in list_of_dicts for key in d.keys())
    dict_of_lists = {key: [] for key in all_keys}

    for d in list_of_dicts:
        for key in all_keys:
            dict_of_lists[key].append(d.get(key, default_value))

    return dict_of_lists

def prepare_images_for_wiki(data_path, img_dir):
    """
    This functor prepares images for WIT data
    1. Download images if `_fetch_images` is set to True
    2. Filter and return only examples that have downloaded images
    """
    """
    inputs: wit_data: with train and valid HF Datasets
    """

    USER_AGENT = get_datasets_user_agent()
    num_threads = 32

    from random_user_agent.user_agent import UserAgent
    from random_user_agent.params import SoftwareName, OperatingSystem

    # you can also import SoftwareEngine, HardwareType, SoftwareType, Popularity from random_user_agent.params
    # you can also set number of user agents required by providing `limit` as parameter
    software_names = [SoftwareName.CHROME.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    def random_userAgent():
        # user_agent_list = ["Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        # "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
        # "Mozilla/5.0 (Windows NT 10.0;) Gecko/20100101 Firefox/61.0",
        # "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
        # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
        # "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
        # "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        # "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
        # ]
        return user_agent_rotator.get_random_user_agent()
    
    def fetch_single_image(image_url, image_path, timeout=30, retries=3):
        if os.path.exists(image_path):
            # print("skipping", image_url)
            return True
        for _ in range(retries + 1):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}
                headers['User-Agent'] = random_userAgent()
                response = requests.get(image_url, stream=True, timeout=timeout, headers=headers)
                # print(image_url, response)
                if response:
                    image = PIL.Image.open(response.raw)
                    image = image.convert('RGB')
                    # print('saving image to ', image, image_path)
                    image = image.save(image_path)
                    return True
                else:
                    image = None
            except Exception:
                image = None
        # print("result:", image_url, image)
        return False
    
    def get_images(batch, batch_id, num_threads, timeout=30, retries=3):
        image_ids = batch['wikidata_id']
        image_paths = [
            os.path.join(img_dir, get_dirname_from_img_id(i), f"{i}.jpg") for i in image_ids
        ]
        batch['image_id'] = image_ids
        batch['img_path'] = image_paths

        fetch_single_image_with_args = partial(
            fetch_single_image, timeout=timeout, retries=retries
        )
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch_images = list(
                executor.map(fetch_single_image_with_args, batch["wikipedia_image_url"], image_paths)
            )
        batch["image_downloaded"] = batch_images
        # print(f"Fetch rate {sum(batch_images)/len(batch_images)}")

        return batch
        
    samples = load_jsonl(data_path)
    samples = Dataset.from_dict(list_of_dicts_to_dict_of_lists(samples))
    
    samples = samples.map(
        get_images,
        batched=True,
        batch_size=512,
        with_indices=True,
        num_proc=32,
        fn_kwargs={
            "num_threads": num_threads,
        },
    )
    print("Image download finished.")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="data/wit/images")
    args = parser.parse_args()
    
    title2img = load_jsonl("data/wiki/Wiki6M_ver_1_0_title_only.jsonl")
    title2img = convert_list2dict(title2img, set_key="wikipedia_title")
    
    total_data = []
    data_list = glob(f"{args.data_dir}/*.tsv")
    for data_path in data_list:
        data = pd.read_csv(data_path, delimiter='\t', keep_default_na=False)
        total_data.extend(inverse_cloze_task(data, title2img))
        
    with open(args.save_path, "w") as fout:
        json.dump(total_data, fout, ensure_ascii=False, indent=2)
    
    print(f"Total items: {len(total_data)}")
