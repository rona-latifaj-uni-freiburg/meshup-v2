"""Read-only audit of the five visualized pairs for the presentation guide.
Reproduces supplementary controls, group diagnostics and exact provenance.
Does not rerun training or alter the original report / visual comparison.
"""
from __future__ import annotations
import csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree
from build_densematcher_visuals import ROOT, OUT, EVAL, error_table, score, csv_rows, write_csv
import matplotlib.pyplot as plt

FIELDS = ['deformation_parameterization','epochs','lr','seed','use_sds','use_dino_loss','image_weight',
'partfield_n_buckets','partfield_min_points','partfield_source_to_target_weight','partfield_target_to_source_weight',
'partfield_src_to_tgt_unmatched_weight','target_mesh_chamfer_weight','target_mesh_partfield_chamfer_weight',
 'target_mesh_partfield_chamfer_global_weight','target_mesh_partfield_chamfer_points',
'jacobian_neighbor_smooth_weight','edge_displacement_jump_weight','edge_displacement_jump_threshold',
'edge_displacement_jump_max_weight','edge_stretch_weight','regularize_jacobians_weight',
'partfield_source_labels','partfield_target_labels']


def main():
    original=json.loads((OUT/'provenance.json').read_text())
    rows=csv_rows(EVAL/'fair_comparison_meshup_vs_densematcher_matched_rows.csv')
    selected=csv_rows(OUT/'case_metrics.csv')
    tests=set((ROOT/'external/DenseCorr3D/test_files.txt').read_text().split())
    archive=zipfile.ZipFile(ROOT/'densecorr_test.zip')
    details=[];regions=[];sweep=[];baseline=[];geometry=[];file_records=[]
    for case in original['cases']:
        cat=case['category'];configpath=next(p for p in case['files'] if p.endswith('config.yml'))
        run=(ROOT/configpath).parent;cfg=yaml.safe_load((ROOT/configpath).read_text())
        predpath=next(p for p in case['files'] if '/pairs/' in p)
        archived='densecorr_test/'+predpath.split('/densecorr_test/')[1]
        assert hashlib.sha256(archive.read(archived)).hexdigest()==hashlib.sha256((ROOT/predpath).read_bytes()).hexdigest()
        detail={**{k:case[k] for k in ['category','source_id','target_id','source_vertices','target_vertices','pf_setting','prediction_key']},
                'config_path':configpath,'settings':{k:cfg.get(k,'ABSENT') for k in FIELDS},
                'archive_member':archived,'archive_byte_match':True,
                'source_in_test_split':cat+'/'+case['source_id'] in tests,'target_in_test_split':cat+'/'+case['target_id'] in tests}
        with np.load(ROOT/predpath,allow_pickle=False) as pred:
            detail['archive_metadata']={k:pred[k].item() for k in ['source','target','category','n_verts_source','n_verts_target','n_ev','maxiter']}
        with np.load(run/'partfield_labels/bucket_labels.npz',allow_pickle=False) as pf:
            n=int(pf['n_buckets']);sc=np.bincount(pf['source_labels'],minlength=n);tc=np.bincount(pf['target_labels'],minlength=n)
            active=(sc>=int(cfg['partfield_min_points']))&(tc>=int(cfg['partfield_min_points']))
            detail['partfield_bucket_counts']={'source':sc.tolist(),'target':tc.tolist(),'active_ids':np.flatnonzero(active).tolist(),'source_vertices_in_inactive_buckets':int(sc[~active].sum())}
            saved=json.loads((run/'partfield_labels/bucket_labels.json').read_text())
            for k in range(n):assert saved['counts'][f'bucket_{k:02d}']['active']==bool(active[k])
        detail['bucket_summary_path']=str((run/'partfield_labels/bucket_labels.json').relative_to(ROOT))
        z=np.load(OUT/'meshes'/cat/'correspondence_and_errors.npz')
        sl,tl=z['source_groups'],z['target_groups'];ed,eo=z['densematcher_error'],z['ours_error']
        detail['source_group_count']=len(np.unique(sl));detail['target_group_count']=len(np.unique(tl))
        detail['source_faces']=len(z['source_faces']);detail['target_faces']=len(z['target_faces'])
        table=error_table(z['target'],z['target_faces'],tl)
        mp=cKDTree(z['target']).query(z['source'])[1];scored=score(table[sl,mp]);assert scored['finite_fraction']==1
        baseline.append({'category':cat,'no_deformation_auc':scored['auc'],'no_deformation_err':scored['err_x100'],'same_group':float(np.mean(tl[mp]==sl))})
        for g in np.unique(sl):
            mask=sl==g
            regions.append({'category':cat,'source_group_zero_based':int(g),'groups_txt_line_one_based':int(g)+1,'vertices':int(mask.sum()),'vertex_fraction':float(mask.mean()),
                            'densematcher_err_x100':float(ed[mask].mean()*100),'ours_err_x100':float(eo[mask].mean()*100),
                            'contribution_to_err_reduction_x100':float((ed[mask]-eo[mask]).sum()*100/len(sl)),
                            'densematcher_same_group':float(np.mean(tl[z['densematcher_map'][mask]]==g)),
                            'ours_same_group':float(np.mean(tl[z['ours_induced_map'][mask]]==g))})
        for r in sorted(rows,key=lambda r:r['pf_setting']):
            if (r['category'],r['source_id'],r['target_id'])!=(cat,case['source_id'],case['target_id']):continue
            sweep.append({'category':cat,'pf':r['pf_setting'],'ours_auc':float(r['meshup_auc']),'ours_err':float(r['meshup_err']),
                          'densematcher_auc':float(r['densematcher_auc']),'densematcher_err':float(r['densematcher_err']),'selected_for_figure':r['pf_setting']==case['pf_setting']})
        if cat in ['broccoli','eggplant']:
            for side in ['source','target']:
                v=z[side];lab=z[side+'_groups'];ann=z[side+'_annotated']
                _,s,vt=np.linalg.svd(v-v.mean(0),full_matrices=False);pc=(v-v.mean(0))@vt.T
                geometry.append({'category':cat,'mesh':side,'group_id_pca_correlations_annotated':[float(np.corrcoef(lab[ann],pc[ann,i])[0,1]) for i in range(3)],
                                 'principal_variance_fractions':(s*s/(s*s).sum()).tolist()})
        checks={**case['files']}
        for extra in ['partfield_labels/bucket_labels.npz','partfield_labels/bucket_labels.json']:
            path=run/extra;checks[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for path,sha in checks.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
            file_records.append({'category':cat,'relative_path':path,'absolute_path':str(ROOT/path),'sha256':sha})
        details.append(detail)
    archive.close()
    for file,data in [('case_file_manifest.csv',file_records),('region_diagnostics.csv',regions),('selected_pair_pf_sensitivity.csv',sweep),('no_deformation_diagnostic.csv',baseline)]:
        write_csv(OUT/file,data)
    for file,data in [('presentation_case_details.json',details),('group_geometry_diagnostics.json',geometry)]:
        (OUT/file).write_text(json.dumps(data,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    cats=[r['category'] for r in selected];x=np.arange(len(cats));w=.25
    for j,(label,color) in enumerate([('Undeformed source','#8a929b'),('DenseMatcher report map','#b5651d'),('Our saved deformation','#1b7a72')]):
        for ax,key,bkey in [(axs[0],'auc','no_deformation_auc'),(axs[1],'err','no_deformation_err')]:
            values=[b[bkey] for b in baseline] if j==0 else [float(r[('densematcher_' if j==1 else 'ours_')+key]) for r in selected]
            ax.bar(x+(j-1)*w,values,w,color=color,label=label)
    for ax in axs:
        ax.set_xticks(x);ax.set_xticklabels([c.capitalize() for c in cats]);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0].set(ylim=(0,1.02),ylabel='AUC (higher is better)');axs[1].set_ylabel('Err = 100 × mean normalized error')
    axs[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Additional control: how much was already correct before deformation?',fontsize=13)
    fig.text(.04,.01,'Same five selected pairs, same normalization and scorer. Undeformed source uses geometric nearest-target-vertex matches.\nThis new control was computed for the discussion; it is not an additional row in the original report benchmark.',fontsize=9)
    fig.subplots_adjust(bottom=.19,wspace=.25)
    for ext in ['pdf','png']:fig.savefig(OUT/('05_no_deformation_control.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    print('Audited',len(details),'pairs;',len(file_records),'file hashes; all prediction files match the supplied ZIP.')

if __name__=='__main__':main()
