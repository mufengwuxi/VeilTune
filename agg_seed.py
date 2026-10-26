import json,sys
names=['SAP','PrivTuner','PrivTuner+act','SecP','SecP+act']
for pre in sys.argv[1:]:
    for i in range(15):
        try: L=[json.loads(l) for l in open(f'rl_leak8/logs/{pre}{i}.log',errors='ignore') if l.startswith('{"stage"')]
        except FileNotFoundError: continue
        r={}
        for d in L:
            st=d['stage']
            if st=='baseline_local': r['local']=round(100*d['agree_thetaT'],1)
            if st=='protocol': r['T']=round(100*d['test_thetaT'],1)
            if st.startswith('attack') and '+y' not in st: r[f"K{d['K']}"]=round(100*d['agree_thetaT'],1)
        print(pre, 42+i//5, names[i%5], r)
