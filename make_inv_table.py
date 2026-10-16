import json
d=json.load(open('out/invariance.json'))
K=['D1','CV','G','rc','nLT','kDC','k2','logR']
ROWS=[('Translation',['translation']),('Diagonal scaling',['diag. scaling (v x10)','diag. scaling (w x100)']),
 ('Time rescaling',['time rescaling (x10)']),('Nonlinear in $w$ only',['componentwise w+0.3w^3']),
 ('Nonlinear in $v$ only',['componentwise v+0.3v^3']),('Shear / rotation',['shear (w+0.5v)','rotation 30deg']),
 ('Nonlinear mixing',['mixing (w+0.3v^2)'])]
def f(x):
    if x<0.05: return '0.0'
    return '%.1f'%x if x<100 else '%.0f'%x
lines=[]
for name,cs in ROWS:
    vals=[];arc=0
    for k in K:
        m=0
        for mod in ['FHN','ML-II','ML-I']:
            r=d[f'{mod}|identity']['time'][k]
            for cname in cs:
                m=max(m,100*abs(d[f'{mod}|{cname}']['time'][k]-r)/max(abs(r),1e-9))
        vals.append(m)
    for mod in ['FHN','ML-II','ML-I']:
        r=d[f'{mod}|identity']['arc']['G']
        for cname in cs: arc=max(arc,100*abs(d[f'{mod}|{cname}']['arc']['G']-r)/r)
    lines.append(name+' & '+' & '.join(f(v) for v in vals)+' & '+f(arc)+r' \\')
print('\n'.join(lines))
open('out/inv_table.tex','w').write('\n'.join(lines))
