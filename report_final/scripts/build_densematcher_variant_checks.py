"""Audit saved output variants, regenerate galleries from fitted-mesh cache.
Run after build_densematcher_visuals.py; never selects the best DenseMatcher variant.
"""
from build_densematcher_visuals import *
from plyfile import PlyData


def ply_vertices(path):
    p=PlyData.read(path)
    return np.column_stack([p['vertex'][k] for k in ['x','y','z']])


def variant_figure(cases, pdf):
    names=['Report map','ICP map','Adjoint map','ICP + adjoint','Hungarian + NN fill','Our induced map']
    fig,axs=plt.subplots(len(cases),6,figsize=(18,3.0*len(cases)))
    rows=[]
    for j,name in enumerate(names):axs[0,j].set_title(name,fontsize=11,pad=24)
    for i,c in enumerate(cases):
        p=next(ROOT/path for path in c['prov']['files'] if path.endswith('.npz'))
        key=c['row']['prediction_key'];table=error_table(c['T'],c['tf'],c['tl'])
        with np.load(p,allow_pickle=False) as z:
            keys=[key,key+'_icp',key+'_adjoint',key+'_icp_adjoint']
            maps=[z[k].copy() for k in keys]
            raw=np.full(len(c['S']),-1,dtype=int)
            if key=='surface_map':raw[z['hungarian_target']]=z['hungarian_source']
            else:raw[z['hungarian_source']]=z['hungarian_target']
            mask=raw>=0;filled=raw.copy()
            known=np.flatnonzero(mask)
            filled[~mask]=raw[known[cKDTree(c['S'][mask]).query(c['S'][~mask])[1]]]
            maps.extend([filled,c['ours']]);keys.extend(['hungarian_geometric_nearest_fill','ours_induced'])
        pal=palette(c)
        for j,(mp,k) in enumerate(zip(maps,keys)):
            assert mp.shape==(len(c['S']),) and mp.min()>=0 and mp.max()<len(c['T'])
            sc=score(table[c['sl'],mp]);unique=len(np.unique(mp))
            panel(axs[i,j],c,c['S'],c['sf'],pal[c['tl'][mp]])
            axs[i,j].text(.5,-.025,f"AUC {sc['auc']:.3f}\n{unique}/{len(c['T'])} target vertices hit",ha='center',va='top',fontsize=9,transform=axs[i,j].transAxes)
            rr={'category':c['cat'],'variant':k,'auc':sc['auc'],'err_x100':sc['err_x100'],'target_vertices_hit':unique,'target_vertices_total':len(c['T']),
                'pre_fill_source_coverage':float(mask.mean()) if j==4 else 1.,
                'matched_only_auc':score(table[c['sl'][mask],mp[mask]])['auc'] if j in [4,5] else '',
                'matched_only_err':score(table[c['sl'][mask],mp[mask]])['err_x100'] if j in [4,5] else ''}
            rows.append(rr)
        axs[i,0].text(0,1.04,c['cat'].capitalize(),weight='bold',transform=axs[i,0].transAxes)
    fig.suptitle('Sensitivity to the supplied DenseMatcher output variant',fontsize=17,y=.995)
    fig.text(.04,.005,'Every column paints predicted target groups onto the same source. Report map = the surface_map / surface_map_inv used in the report.\nHungarian uses stored one-to-one matches; unmatched source vertices inherit the nearest matched source vertex’s target index (geometric NN fill).\nTarget-vertex hit count measures discrete map concentration, not area coverage or semantic accuracy. No variant is selected as the winner.',fontsize=10)
    fig.subplots_adjust(top=.94,bottom=.085,hspace=.29,wspace=.04)
    write_csv(OUT/'densematcher_variant_metrics.csv',rows)
    save_figure(fig,'04_densematcher_output_variants',pdf)
    return rows


def main():
    prov=json.loads((OUT/'provenance.json').read_text());cases=[]
    for row in choose_cases():
        c=load_case(row);dest=OUT/'meshes'/c['cat']
        cp=next(p for p in prov['cases'] if p['category']==c['cat'])
        c['fits']={};c['fit_metrics']={}
        for name in ['densematcher','ours']:
            for lam in LAMBDAS:
                c['fits'][(name,lam)]=ply_vertices(dest/f'{name}_arap_lambda_{lam:g}.ply')
                c['fit_metrics'][(name,lam)]=cp['arap'][f'{name}_lambda_{lam:g}']
        cases.append(c)
    import sys
    if '--variant-only' in sys.argv:
        variant_figure(cases,None)
        return
    with PdfPages(OUT/'visual_comparison.pdf') as pdf:
        semantic_figure(cases,pdf);error_figure(cases,pdf);deformation_figure(cases,pdf)
        rows=variant_figure(cases,pdf)
        for c in cases:detail_figures(c,pdf)
    print('Output variants:',[(r['category'],r['variant'],round(r['auc'],3)) for r in rows],flush=True)

if __name__=='__main__':main()
