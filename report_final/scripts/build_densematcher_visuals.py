"""Reproducible, CPU-only visual comparison of saved DenseMatcher and pipeline outputs.

Run with the meshup_new environment. Writes only report_final/densematcher_visuals.
No learning, ground-truth-guided fitting, remeshing, or per-method camera changes.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/meshup_visuals_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import csv, json, hashlib, re
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, eye, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import factorized
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report_final/densematcher_visuals'
EVAL = ROOT / 'jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy'
DATA = ROOT / 'external/DenseCorr3D'
CATEGORIES = ['animals', 'cars', 'chairs', 'broccoli', 'eggplant']
LAMBDAS = [0.1, 1.0, 10.0]
CAMERAS = {'animals': (25, 15), 'cars': (45, 18), 'chairs': (35, 18), 'broccoli': (90, 15), 'eggplant': (90, 15)}
DM_COLOR, OUR_COLOR = '#b5651d', '#1b7a72'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': '#20242c', 'axes.labelcolor': '#20242c', 'pdf.fonttype': 42})


def csv_rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def obj(path):
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if not tok: continue
        if tok[0] == 'v': v.append([float(x) for x in tok[1:4]])
        elif tok[0] == 'f':
            ids = [int(x.split('/')[0]) for x in tok[1:]]
            ids = [i - 1 if i > 0 else len(v) + i for i in ids]
            assert len(ids) == 3, 'Require original triangular topology'
            f.append(ids)
    return np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64)


def unit(v):
    return (v - (v.min(0) + v.max(0)) / 2) * (2 / np.ptp(v, axis=0).max())


def edges(f):
    return np.unique(np.sort(np.concatenate([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]]), axis=1), axis=0)


def labels(path, v):
    # Exactly the report's inference/fill policy; retain annotation mask separately.
    groups = [[int(x) for x in re.findall(r'-?\d+', l.split('#')[0])] for l in Path(path).read_text().splitlines()]
    groups = [g for g in groups if g]
    values = np.concatenate(groups)
    base = 1 if values.min() >= 1 and values.max() == len(v) else 0
    lab = np.full(len(v), -1, dtype=int)
    for k, g in enumerate(groups):
        ids = np.asarray(g) - base
        assert ids.min() >= 0 and ids.max() < len(v)
        lab[ids] = k
    mask = lab >= 0
    assigned = np.flatnonzero(mask)
    # float32 exhaustive nearest fill matches densecorr3d_segment.py, including ties.
    for i in np.flatnonzero(~mask):
        d = v[i].astype(np.float32) - v[assigned].astype(np.float32)
        lab[i] = lab[assigned[np.argmin(np.einsum('ij,ij->i', d, d))]]
    return lab, mask


def error_table(v, f, lab, annotated=None):
    ed = edges(f); i,j = ed.T
    w = np.linalg.norm(v[i]-v[j], axis=1)
    graph = coo_matrix((np.r_[w,w], (np.r_[i,j], np.r_[j,i])), shape=(len(v),len(v))).tocsr()
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
    table = np.full((lab.max()+1,len(v)),np.inf)
    for k in np.unique(lab):
        ids = np.flatnonzero((lab == k) & (True if annotated is None else annotated))
        if len(ids): table[k] = dijkstra(graph,directed=False,indices=ids,min_only=True)/np.sqrt(area)
    return table


def score(err):
    finite = err[np.isfinite(err)]
    ts = np.linspace(0,.25,251)
    pck = np.array([(finite <= t).mean() for t in ts])
    return {'auc': float(np.trapz(pck,ts)/.25), 'err_x100': float(finite.mean()*100), 'finite_fraction': float(np.isfinite(err).mean()), 'pck': pck}


def choose_cases():
    rows = csv_rows(EVAL/'fair_comparison_meshup_vs_densematcher_matched_rows.csv')
    chosen, selection = [], []
    for cat in CATEGORIES:
        allrows = [r for r in rows if r['category']==cat]
        levels = sorted({r['pf_setting'] for r in allrows})
        pf = levels[(len(levels)-1)//2]
        candidates = [r for r in allrows if r['pf_setting']==pf]
        candidates.sort(key=lambda r:(float(r['meshup_auc'])-float(r['densematcher_auc']),r['source_id'],r['target_id']))
        rank = (len(candidates)-1)//2
        chosen.append(candidates[rank])
        for j,r in enumerate(candidates):
            selection.append({**r,'delta_auc':float(r['meshup_auc'])-float(r['densematcher_auc']), 'rank':j+1,'selected':j==rank})
    write_csv(OUT/'selection_candidates.csv', selection)
    return chosen


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def load_case(row):
    cat, sid, tid = row['category'],row['source_id'],row['target_id']
    raw = next(r for r in csv_rows(EVAL/row['csv_source']) if r['run_name']==row['run_name'])
    final_path=ROOT/raw['deformed_mesh'];run=final_path.parent.parent
    cfg=yaml.safe_load((run/'config.yml').read_text())
    sv,sf=obj(DATA/cat/sid/'simple_mesh.obj');tv,tf=obj(DATA/cat/tid/'simple_mesh.obj')
    cv,cf=obj(ROOT/(cfg.get('staged_original_source_mesh') or cfg['mesh']))
    ctv,ctf=obj(ROOT/cfg['target_mesh'])
    assert np.allclose(sv,cv,atol=1e-7) and np.array_equal(sf,cf), 'Source preparation changed; needs explicit transport'
    assert np.allclose(tv,ctv,atol=1e-7) and np.array_equal(tf,ctf), 'Target preparation changed; needs explicit transport'
    fv,ff=obj(final_path)
    assert fv.shape==sv.shape and np.array_equal(sf,ff), 'Source topology/order not retained'
    # Check initial logged positions too, not only vertex counts.
    from plyfile import PlyData
    initial=run/'colored_meshes/mesh_initial.ply'
    ply=PlyData.read(initial)
    iv=np.column_stack([ply['vertex'][k] for k in ['x','y','z']])
    iff=np.stack(ply['face']['vertex_indices'])
    assert np.array_equal(iff,sf) and np.allclose(unit(iv),unit(sv),atol=2e-5), 'Initial vertex order changed'
    sl,sa=labels(DATA/cat/sid/'groups.txt',sv);tl,ta=labels(DATA/cat/tid/'groups.txt',tv)
    S,T,X=unit(sv),unit(tv),unit(fv)
    ours=cKDTree(T).query(X,workers=1)[1]
    found=[]
    for p in sorted((DATA/'densecorr_test/pairs'/cat).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            src=str(z['source'].item()).split('/')[-1];tgt=str(z['target'].item()).split('/')[-1]
            if (src,tgt)==(sid,tid): key='surface_map_inv'
            elif (src,tgt)==(tid,sid): key='surface_map'
            else: continue
            if key != row['prediction_key']: continue
            found.append((p,key,z[key].astype(int).reshape(-1)))
    assert len(found)==1, f'Expected unambiguous directed map, found {len(found)}'
    pair_path,key,dm=found[0]
    assert dm.shape==(len(S),) and dm.min()>=0 and dm.max()<len(T)
    table=error_table(T,tf,tl)
    ed,eo=table[sl,dm],table[sl,ours]
    sd,so=score(ed),score(eo)
    for m,computed in [('densematcher',sd),('meshup',so)]:
        assert abs(computed['auc']-float(row[m+'_auc']))<2e-7, (cat,m,computed['auc'],row[m+'_auc'])
        assert abs(computed['err_x100']-float(row[m+'_err']))<2e-5
    # Completeness and annotated-only sensitivity are stored, never hidden by a color cap.
    ann=error_table(T,tf,tl,ta)
    annotated_scores={}
    for name,mp in [('densematcher',dm),('ours',ours)]:
        se=score(ann[sl[sa],mp[sa]])
        annotated_scores[name]={k:v for k,v in se.items() if k!='pck'}
    paths=[final_path,run/'config.yml',initial,pair_path,DATA/cat/sid/'simple_mesh.obj',DATA/cat/tid/'simple_mesh.obj',DATA/cat/sid/'groups.txt',DATA/cat/tid/'groups.txt']
    prov={'category':cat,'source_id':sid,'target_id':tid,'pf_setting':row['pf_setting'],'prediction_key':key,
          'source_vertices':len(S),'target_vertices':len(T),'source_annotation_fraction':float(sa.mean()),'target_annotation_fraction':float(ta.mean()),
          'source_topology_and_initial_order_verified':True,'scores_reproduced':True,'annotated_only_sensitivity':annotated_scores,
          'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'run_name':row['run_name']}
    return dict(row=row,cat=cat,S=S,T=T,X=X,sf=sf,tf=tf,sl=sl,tl=tl,sa=sa,ta=ta,dm=dm,ours=ours,ed=ed,eo=eo,sd=sd,so=so,prov=prov)


def arap(source, faces, anchors, lam, iterations=100):
    """Uniform-weight symmetric vertex-spokes ARAP, all vertices soft constrained.
    E = mean_i ||X_i-A_i||^2 + lambda/(2|E|) sum_{ij in E}
       (||X_i-X_j-R_i(S_i-S_j)||^2 + ||X_i-X_j-R_j(S_i-S_j)||^2).
    The same initial source, schedule and weights are used for both maps.
    """
    ed=edges(faces);i,j=ed.T;n=len(source);m=len(ed)
    adj=coo_matrix((np.ones(2*m),(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    lap=diags(np.asarray(adj.sum(axis=1)).ravel())-adj
    alpha=lam*n/m
    solve=factorized((eye(n,format='csc')+alpha*lap).tocsc())
    p=source[i]-source[j];X=source.copy();hist=[]
    def rotations(x):
        q=x[i]-x[j]
        cov=np.zeros((n,3,3));outer=q[:,:,None]*p[:,None,:]
        np.add.at(cov,i,outer);np.add.at(cov,j,outer)
        u,_,vt=np.linalg.svd(cov)
        sign=np.ones((n,3));sign[:,-1]=np.linalg.det(u@vt)
        return (u*sign[:,None,:])@vt
    R=rotations(X)
    for step in range(iterations):
        rp=np.einsum('nij,nj->ni',(R[i]+R[j])/2,p)
        b=np.zeros_like(X);np.add.at(b,i,rp);np.add.at(b,j,-rp)
        nxt=np.column_stack([solve(anchors[:,d]+alpha*b[:,d]) for d in range(3)])
        delta=float(np.max(np.linalg.norm(nxt-X,axis=1)))
        X=nxt;R=rotations(X);q=X[i]-X[j]
        e=float(np.mean(np.sum((X-anchors)**2,axis=1))+lam/(2*m)*(np.sum((q-np.einsum('nij,nj->ni',R[i],p))**2)+np.sum((q-np.einsum('nij,nj->ni',R[j],p))**2)))
        hist.append(e)
        if delta<1e-7: break
    assert np.isfinite(X).all()
    assert np.all(np.diff(hist)<=1e-9), 'ARAP energy should descend'
    return X,{'iterations':len(hist),'energy':hist[-1],'last_vertex_step':delta,'energy_monotone':True,'energy_history':hist}


def solver_checks():
    s=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    f=np.array([[0,1,2],[0,2,3]])
    x,h=arap(s,f,s,1);assert np.max(abs(x-s))<1e-10
    x,h=arap(s,f,s+[.2,-.4,.7],1);assert np.max(abs(x-(s+[.2,-.4,.7])))<1e-8
    theta=.3;rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    a=s@rot.T+[.2,-.4,.7];x,h=arap(s,f,a,1,300);assert np.max(abs(x-a))<2e-6
    return {'identity':True,'translation':True,'rigid_rotation':True,'energy_descent':True}


def deform_metrics(s,f,x,anchors):
    ed=edges(f);length0=np.linalg.norm(s[ed[:,0]]-s[ed[:,1]],axis=1)
    length=np.linalg.norm(x[ed[:,0]]-x[ed[:,1]],axis=1)
    valid=length0>1e-12
    stretch=length[valid]/length0[valid]
    tri=x[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    src=s[f];area0=np.linalg.norm(np.cross(src[:,1]-src[:,0],src[:,2]-src[:,0]),axis=1)/2
    validface=area0>1e-12
    return {'anchor_rmse':float(np.sqrt(np.mean(np.sum((x-anchors)**2,axis=1)))),
            'edge_stretch_p95':float(np.percentile(stretch,95)),'edge_stretch_p05':float(np.percentile(stretch,5)),
            'near_collapsed_face_fraction':float(np.mean(area[validface]/area0[validface]<.01))}


def write_ply(path,v,f,c):
    # ASCII vertex colors make artifacts usable in MeshLab/Blender, keep vertex IDs.
    c=np.clip(c*255,0,255).astype(int)
    with Path(path).open('w') as o:
        o.write(f'ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p,rgb in zip(v,c): o.write(' '.join(map(str,[*p,*rgb]))+'\n')
        for tri in f: o.write('3 '+' '.join(map(str,tri))+'\n')


def palette(c):
    n=max(c['sl'].max(),c['tl'].max())+1
    if n<=10: return plt.get_cmap('tab10')(np.arange(n))[:,:3]
    # Ordered ring-like annotations (e.g. eggplant) get a continuous indexed palette.
    return plt.get_cmap('turbo')(np.linspace(.03,.97,n))[:,:3]


def rotate(v, az, el):
    a,e=np.deg2rad([az,el]);ca,sa,ce,se=np.cos(a),np.sin(a),np.cos(e),np.sin(e)
    x=ca*v[:,0]+sa*v[:,2];z=-sa*v[:,0]+ca*v[:,2];y=v[:,1]
    return np.column_stack([x,ce*y-se*z,se*y+ce*z])


def frame(c,az,el):
    geometries=[c[k] for k in ['S','T','X']]+list(c.get('fits',{}).values())
    points=np.concatenate([rotate(v,az,el) for v in geometries])
    # Same source/target/actual-output frame for every method in the row.
    lo,hi=points[:,:2].min(0),points[:,:2].max(0)
    return (lo+hi)/2,max((hi-lo)[0]/1.22,(hi-lo)[1])*1.10


def render(v,f,colors,az,el,bounds,width=550,height=450,shade=True):
    """Orthographic CPU triangle rasterizer with barycentric color and real z-buffer."""
    vv=rotate(v,az,el);center,span=bounds;scale=height/span
    screen=vv[:,:2].copy();screen[:,0]=(screen[:,0]-center[0])*scale+width/2
    screen[:,1]=-(screen[:,1]-center[1])*scale+height/2
    rgb=np.ones((height,width,3),float);depth=np.full((height,width),-np.inf)
    tri=vv[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-12)
    light=np.array([.25,.45,.86]);light/=np.linalg.norm(light)
    brightness=.78+.22*abs(norm@light) if shade else np.ones(len(f))
    for k,ids in enumerate(f):
        p=screen[ids];lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int),[width-1,height-1])
        if np.any(hi<lo):continue
        x0,y0=p[0];x1,y1=p[1];x2,y2=p[2]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-9:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
        a=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
        b=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;d=1-a-b
        z=a*vv[ids[0],2]+b*vv[ids[1],2]+d*vv[ids[2],2]
        patch=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];inside=(a>=-1e-8)&(b>=-1e-8)&(d>=-1e-8)&(z>patch)
        if not inside.any():continue
        color=(a[...,None]*colors[ids[0]]+b[...,None]*colors[ids[1]]+d[...,None]*colors[ids[2]])*brightness[k]
        rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1][inside]=color[inside];patch[inside]=z[inside]
    return np.uint8(np.clip(rgb*255,0,255))


def panel(ax,c,v,f,colors,az=None,el=None,shade=True):
    a,e=CAMERAS[c['cat']]
    if az is not None:a=az
    if el is not None:e=el
    ax.imshow(render(v,f,colors,a,e,frame(c,a,e),shade=shade))
    ax.set_axis_off()


def save_figure(fig,name,pdf=None):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    if pdf:pdf.savefig(fig,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Saved',name,flush=True)


def semantic_figure(cases,pdf):
    fig,axs=plt.subplots(len(cases),5,figsize=(16,3.0*len(cases)))
    titles=['Source: reference groups','DenseMatcher (report map)\npredicted groups on source','Ours: predicted groups\non unchanged source','Ours: actual deformation\nsource colors carried along','Target: reference groups']
    for j,t in enumerate(titles):axs[0,j].set_title(t,fontsize=11,pad=24)
    for i,c in enumerate(cases):
        pal=palette(c);sl,tl=c['sl'],c['tl']
        specs=[('S','sf',pal[sl]),('S','sf',pal[tl[c['dm']]]),('S','sf',pal[tl[c['ours']]]),('X','sf',pal[sl]),('T','tf',pal[tl])]
        for j,(v,f,col) in enumerate(specs):panel(axs[i,j],c,c[v],c[f],col)
        axs[i,0].text(0,1.04,f"{c['cat'].capitalize()}  |  {c['row']['pf_setting']}",transform=axs[i,0].transAxes,weight='bold')
        for j,sc in [(1,c['sd']),(2,c['so'])]:
            axs[i,j].text(.5,-.025,f"AUC {sc['auc']:.3f}  |  Err {sc['err_x100']:.2f}",ha='center',transform=axs[i,j].transAxes,color=DM_COLOR if j==1 else OUR_COLOR)
    fig.suptitle('Do the same source regions map to the correct target regions?',fontsize=17,y=.995)
    fig.text(.04,.005,'Colors = held-out DenseCorr3D groups (consistent within each row); predicted color at source vertex i = target group at its predicted match.\nEach row uses one shared camera and scale. Single-PF case studies; these are not the category-averaged headline results.',fontsize=10)
    fig.subplots_adjust(top=.94,bottom=.065,hspace=.24,wspace=.03)
    save_figure(fig,'01_semantic_correspondence',pdf)


def error_figure(cases,pdf):
    fig,axs=plt.subplots(len(cases),3,figsize=(12,3.0*len(cases)),gridspec_kw={'width_ratios':[1,1,1.05]})
    norm=Normalize(0,.25,clip=True);cmap=plt.get_cmap('magma')
    for j,t in enumerate(['DenseMatcher error','Our induced-map error','All source vertices (including hidden side)']):axs[0,j].set_title(t,fontsize=11,pad=24)
    for i,c in enumerate(cases):
        for j,err in enumerate([c['ed'],c['eo']]):
            col=cmap(norm(err))[:,:3];col[~np.isfinite(err)]=[0,1,1]
            panel(axs[i,j],c,c['S'],c['sf'],col,shade=False)
        axs[i,0].text(0,1.02,f"{c['cat'].capitalize()} | {c['row']['pf_setting']}",weight='bold',transform=axs[i,0].transAxes)
        ax=axs[i,2];ts=np.linspace(0,.25,251)
        for sc,color,name in [(c['sd'],DM_COLOR,'DenseMatcher'),(c['so'],OUR_COLOR,'Ours')]:
            ax.plot(ts,sc['pck'],color=color,lw=2,label=f"{name}: AUC {sc['auc']:.3f}")
        ax.set(xlim=(0,.25),ylim=(0,1.02),xlabel='Normalized graph-geodesic threshold',ylabel='Fraction correct')
        ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Where does correspondence fail?',fontsize=17,y=.995)
    fig.subplots_adjust(top=.94,bottom=.085,wspace=.26,hspace=.34)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs[:,:2].ravel().tolist(),location='bottom',fraction=.015,pad=.03,extend='max')
    cb.set_label('Normalized semantic graph-geodesic error\nShared scale; values above 0.25 are saturated',fontsize=10)
    save_figure(fig,'02_error_maps_and_curves',pdf)


def deformation_figure(cases,pdf):
    fig,axs=plt.subplots(len(cases),5,figsize=(16,3.0*len(cases)))
    titles=['Source','DenseMatcher + ARAP\nnew downstream reconstruction','Our induced map + ARAP\nsame reconstruction control','Our actual output\nsaved PartField deformation','Target']
    for j,t in enumerate(titles):axs[0,j].set_title(t,fontsize=11,pad=24)
    for i,c in enumerate(cases):
        pal=palette(c);col=pal[c['sl']]
        specs=[(c['S'],c['sf'],col),(c['fits'][('densematcher',1.)],c['sf'],col),(c['fits'][('ours',1.)],c['sf'],col),(c['X'],c['sf'],col),(c['T'],c['tf'],pal[c['tl']])]
        for j,(v,f,color) in enumerate(specs):panel(axs[i,j],c,v,f,color)
        axs[i,0].text(0,1.04,c['cat'].capitalize()+' | '+c['row']['pf_setting'],weight='bold',transform=axs[i,0].transAxes)
    fig.suptitle('What deformation do the correspondences support?',fontsize=17,y=.995)
    fig.text(.04,.005,'ARAP uses identical source initialization, all-vertex soft anchors, uniform edge weights and lambda = 1 for both maps.\nDenseMatcher does not natively output this deformation. Source colors track vertex identity; the target uses its own reference groups. See all three weights in the sensitivity pages.',fontsize=10)
    fig.subplots_adjust(top=.94,bottom=.065,hspace=.24,wspace=.03)
    save_figure(fig,'03_controlled_deformation',pdf)


def detail_figures(c,pdf):
    cat=c['cat'];pal=palette(c);col=pal[c['sl']]
    fig,axs=plt.subplots(2,4,figsize=(14,7.5))
    az,el=CAMERAS[cat]
    for i,a in enumerate([az,az+180]):
        for j,(v,co) in enumerate([(c['S'],pal[c['tl'][c['dm']]]),(c['S'],pal[c['tl'][c['ours']]]),(c['fits'][('densematcher',1.)],col),(c['X'],col)]):
            panel(axs[i,j],c,v,c['sf'],co,az=a,el=el)
            axs[i,j].set_title(['DenseMatcher: predicted groups','Ours: predicted groups','DenseMatcher + ARAP (lambda = 1)','Ours: actual deformation'][j],fontsize=10)
        axs[i,0].text(0,1.08,'View '+str(i+1),transform=axs[i,0].transAxes,weight='bold')
    fig.suptitle(f"{cat.capitalize()} | {c['row']['pf_setting']} | opposite-view check",fontsize=15)
    fig.text(.05,.015,f"Source {c['row']['source_id']}  →  Target {c['row']['target_id']}\nGroup colors are for evaluation/visualization only; ARAP anchors contain no group labels.",fontsize=9)
    fig.subplots_adjust(bottom=.10,hspace=.18)
    save_figure(fig,f'{cat}_opposite_views',pdf)
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for i,name in enumerate(['densematcher','ours']):
        for j,lam in enumerate(LAMBDAS):
            panel(axs[i,j],c,c['fits'][(name,lam)],c['sf'],col)
            met=c['fit_metrics'][(name,lam)]
            axs[i,j].set_title(f"{'DenseMatcher' if i==0 else 'Our induced map'} + ARAP | lambda {lam:g}",fontsize=11)
            axs[i,j].text(.5,-.03,f"Anchor RMSE {met['anchor_rmse']:.3f} | edge stretch p95 {met['edge_stretch_p95']:.2f}",ha='center',transform=axs[i,j].transAxes,fontsize=9)
    fig.suptitle(f'{cat.capitalize()}: identical reconstruction, three fixed regularization weights',fontsize=15)
    fig.text(.05,.015,'Source colors are carried by vertex index. Geometry is never re-normalized after ARAP. Larger lambda favors local rigidity.\nThese are constructed downstream fits, not native DenseMatcher meshes. No weight was chosen using ground-truth scores.',fontsize=10)
    fig.subplots_adjust(bottom=.10,hspace=.18)
    save_figure(fig,f'{cat}_arap_sensitivity',pdf)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=solver_checks();print('ARAP correctness checks passed',flush=True)
    cases=[];metrics=[]
    for row in choose_cases():
        c=load_case(row);cases.append(c)
        print(c['cat'],'metrics reproduced',c['sd']['auc'],c['so']['auc'],flush=True)
        dest=OUT/'meshes'/c['cat'];dest.mkdir(parents=True,exist_ok=True)
        pal=palette(c);col=pal[c['sl']]
        write_ply(dest/'source.ply',c['S'],c['sf'],col)
        write_ply(dest/'target.ply',c['T'],c['tf'],pal[c['tl']])
        write_ply(dest/'ours_actual.ply',c['X'],c['sf'],col)
        np.savez_compressed(dest/'correspondence_and_errors.npz',source=c['S'],target=c['T'],source_faces=c['sf'],target_faces=c['tf'],ours_actual=c['X'],densematcher_map=c['dm'],ours_induced_map=c['ours'],densematcher_error=c['ed'],ours_error=c['eo'],source_groups=c['sl'],target_groups=c['tl'],source_annotated=c['sa'],target_annotated=c['ta'])
        c['fits']={};c['fit_metrics']={};c['prov']['arap']={}
        for name,mp in [('densematcher',c['dm']),('ours',c['ours'])]:
            for lam in LAMBDAS:
                fit,h=arap(c['S'],c['sf'],c['T'][mp],lam)
                c['fits'][(name,lam)]=fit
                met=deform_metrics(c['S'],c['sf'],fit,c['T'][mp]);c['fit_metrics'][(name,lam)]=met
                c['prov']['arap'][f'{name}_lambda_{lam:g}']={**h,**met}
                write_ply(dest/f'{name}_arap_lambda_{lam:g}.ply',fit,c['sf'],col)
                metrics.append({'category':c['cat'],'method':name,'lambda':lam,**met,'iterations':h['iterations'],'last_vertex_step':h['last_vertex_step']})
        print(c['cat'],'ARAP fits complete',flush=True)
    write_csv(OUT/'arap_metrics.csv',metrics)
    summary=[]
    for c in cases:
        summary.append({'category':c['cat'],'pf':c['row']['pf_setting'],'source_id':c['row']['source_id'],'target_id':c['row']['target_id'],
                        'densematcher_auc':c['sd']['auc'],'ours_auc':c['so']['auc'],'densematcher_err':c['sd']['err_x100'],'ours_err':c['so']['err_x100'],
                        'densematcher_same_group':float(np.mean(c['tl'][c['dm']]==c['sl'])),'ours_same_group':float(np.mean(c['tl'][c['ours']]==c['sl'])),
                        'source_annotated_fraction':float(c['sa'].mean()),'target_annotated_fraction':float(c['ta'].mean())})
    write_csv(OUT/'case_metrics.csv',summary)
    (OUT/'provenance.json').write_text(json.dumps({'selection':'Five declared clean-preparation categories. Lower middle tested PF; lower median directed-pair AUC improvement at that PF. Pair selection uses scores for typicality, not an unbiased sample.','solver_checks':checks,'cases':[c['prov'] for c in cases]},indent=2))
    with PdfPages(OUT/'visual_comparison.pdf') as pdf:
        semantic_figure(cases,pdf);error_figure(cases,pdf);deformation_figure(cases,pdf)
        for c in cases:detail_figures(c,pdf)
    print('Complete:',OUT,flush=True)

if __name__=='__main__':main()
