from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
F={'mrpc':('nyu-mll/glue',['mrpc/train-00000-of-00001.parquet','mrpc/validation-00000-of-00001.parquet']),
   'rte':('nyu-mll/glue',['rte/train-00000-of-00001.parquet','rte/validation-00000-of-00001.parquet']),
   'yelp':('fancyzhx/yelp_polarity',['plain_text/train-00000-of-00001.parquet','plain_text/test-00000-of-00001.parquet']),
   'agnews':('fancyzhx/ag_news',['data/train-00000-of-00001.parquet','data/test-00000-of-00001.parquet'])}
for k,(r,fs) in F.items():
    for f in fs:
        p=hf_hub_download(r,f,repo_type='dataset',local_dir='.')
        t=pq.read_table(p); print(k,f,t.num_rows,t.column_names, flush=True)
print('DL_DONE')
