import json,glob,re,os
print('== gate')
for s in (42,43,44):
    f=f'sec_train_adam/sst2_s{s}_gate.log'
    if not os.path.exists(f): continue
    for l in open(f):
        if l.startswith('T16 '):
            d=json.loads(l[4:]); sk=d.get('skipped'); print(s, d['step'], round(d['test_acc']*100,1), round(d['plain_test_acc']*100,1), round(d['rel_dist_to_plain'],4), (sum(sk),len(sk)) if sk else '')
    if 'Traceback' in open(f).read() or 'out of memory' in open(f).read(): print('ERR', f)
print('== leak', 'done' if os.path.exists('rl_leak8/done') else '', 'done_ag' if os.path.exists('rl_leak8/done_ag') else '')
for f in sorted(glob.glob('rl_leak8/logs/j*.log')+glob.glob('rl_leak8/logs/ag*.log'), key=lambda x:(x[14:16],int(re.findall(r'\d+',x)[-1]))):
    txt=open(f).read(); out=[]
    if 'Traceback' in txt or 'out of memory' in txt: out.append('ERR')
    for l in txt.splitlines():
        if not l.startswith('{"stage"'): continue
        d=json.loads(l); st=d['stage']
        if st=='protocol': out.append('T%.1f'%(d['test_thetaT']*100))
        elif st=='baseline_local': out.append('L%.1f/%.1f'%(d['test']*100,d['agree_thetaT']*100))
        elif st.startswith('attack') and '+y' not in st: out.append('K%d:%.1f/%.1f'%(d['K'],d['test']*100,d['agree_thetaT']*100))
    print(f.split('/')[-1], ' '.join(out))
