import json,sys
pre = sys.argv[1] if len(sys.argv)>1 else 'j'
names=['SAP','Pencil','Pencil+pub','SecP','SecP+pub']
R={n:{} for n in names}
for i in range(15):
    n=names[i%5]
    try: L=[json.loads(l) for l in open(f'rl_leak8/logs/{pre}{i}.log') if l.startswith('{"stage"')]
    except FileNotFoundError: continue
    for d in L:
        st=d['stage']
        if st=='protocol': R[n].setdefault('T',[]).append(d['test_thetaT'])
        if st=='baseline_local': R[n].setdefault('Lt',[]).append(d['test']); R[n].setdefault('La',[]).append(d['agree_thetaT'])
        if st.startswith('attack') and '+y' not in st:
            R[n].setdefault(f"K{d['K']}t",[]).append(d['test']); R[n].setdefault(f"K{d['K']}a",[]).append(d['agree_thetaT'])
for n in names:
    print(n, ' '.join(f"{k}={100*sum(v)/len(v):.1f}(n{len(v)})" for k,v in R[n].items()))
