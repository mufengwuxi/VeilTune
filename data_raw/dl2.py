from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
for r,f in [('fancyzhx/ag_news','data/test-00000-of-00001.parquet'),('fancyzhx/ag_news','data/train-00000-of-00001.parquet'),
            ('fancyzhx/yelp_polarity','plain_text/test-00000-of-00001.parquet'),('fancyzhx/yelp_polarity','plain_text/train-00000-of-00001.parquet')]:
    try:
        p=hf_hub_download(r,f,repo_type='dataset',local_dir='.'); t=pq.read_table(p); print('OK',r,f,t.num_rows,flush=True)
    except Exception as e: print('FAIL',r,f,repr(e)[:200],flush=True)
print('DL2_DONE')
