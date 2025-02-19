from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, argparse

from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.indexing import XknowSearcher
from dataset.vqa_ret import get_collection

xknow_ckpt="ckpts/mire/okvqa_gs/base/xknow_epoch_14.pth"
all_blocks_file="data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/okvqa_full_clean_corpus.csv"
index_name="okvqa_gs.nbits=2"
collection = get_collection(all_blocks_file)
config = ColBERTConfig(nbits=2, doc_maxlen=384, query_maxlen=64)
with Run().context(RunConfig(experiment=index_name.split(".")[0])):
    searcher = XknowSearcher(index=index_name, checkpoint=xknow_ckpt, config=config, collection=collection)
    
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    search_text = request.form.get('searchText', '')
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        rank = searcher.search(search_text, save_path, k=10)
        search_results = []
        for passage_id, passage_rank, passage_score in zip(*rank):
            search_results.append({
                'rank': passage_rank,
                'score': passage_score,
                'text': searcher.collection[passage_id]
            })
            
        return jsonify({
            'results': search_results
        })

if __name__ == '__main__':
    app.run(debug=True, port=8024)
