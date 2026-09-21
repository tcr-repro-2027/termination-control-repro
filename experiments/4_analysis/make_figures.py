#!/usr/bin/env python3
"""Render paper figures from the supplied results, without running experiments.

Inputs below --results:
  R0/F1_example_with_text.json; input/e1_metrics.csv;
  R0/R0_pair_metrics.csv;
  R0/*/p0d_episode_hazard/P0D_EPISODE_SUMMARY.csv;
  R1/R1_paired_effects_4B.csv; R1/R1_paired_effects_8B.csv;
  R2/R2_short_effects_4B.csv; R2/R2_short_effects_8B.csv.
Usage:
  python experiments/4_analysis/make_figures.py --results reference_results --output out/figures
  python experiments/4_analysis/make_figures.py --results $PAPER_ROOT --metrics $EVAL_ROOT/e1_metrics.csv
--results is a result tree (reference_results/ or $PAPER_ROOT).  When it has no
input/e1_metrics.csv, --metrics (default $EVAL_ROOT/e1_metrics.csv) supplies that table.
Optional --previews DIR also writes PNG copies.
Figures are exactly 5.5 inches wide. No TeX engine is used.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

COL = {'cleaned':'#3E6FB0','raw':'#D2694A','OBR':'#7B5EA7',
       'untuned':'#8A8A8A','control':'#4F9D69','random':'#B0B0B0'}
plt.rcParams.update({'text.usetex':False,'font.family':'DejaVu Sans','font.size':7.5,
 'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'legend.frameon':False,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'lines.linewidth':1.15,'lines.markersize':3,'axes.linewidth':.55,
 'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2})
C='qwen3-4b-cleanv2-s42';K='qwen3-4b-keep4-s42';O='qwen3-4b-obr-p15-s42'
C8='qwen3-8b-cleanv2';K8='qwen3-8b-keep4-s42'

class Data:
 def __init__(self,root:Path,metrics:Path|None=None):self.root=root;self.metrics=metrics;self.cache={}
 def csv(self,path:str)->pd.DataFrame:
  if path not in self.cache:
   file=self.root/path
   if path=='input/e1_metrics.csv' and not file.is_file() and self.metrics:file=self.metrics
   self.cache[path]=pd.read_csv(file)
  return self.cache[path]
 def one(self,path:str,**filters):
  d=self.csv(path)
  for key,val in filters.items():d=d[d[key].astype(str)==str(val)]
  if len(d)!=1:raise ValueError(f'Expected one row in {path}: {filters}, got {len(d)}')
  return d.iloc[0]

def save(fig,stem,out,previews):
 fig.savefig(out/(stem+'.pdf'),dpi=300,metadata={'Title':stem,'Creator':'matplotlib; source data only'})
 if previews is not None:fig.savefig(previews/(stem+'.png'),dpi=190)
 # Width is independent of artist extents: never use bbox_inches='tight'.
 assert abs(fig.get_size_inches()[0]-5.5)<1e-9
 plt.close(fig)

def style(ax):
 ax.grid(axis='y',alpha=.15,linewidth=.5);ax.set_axisbelow(True)

def err(ax,x,y,lo,hi,**kwargs):
 ax.errorbar(x,y,yerr=np.vstack([np.maximum(0,np.asarray(y)-lo),np.maximum(0,np.asarray(hi)-y)]),
             capsize=1.5,elinewidth=.7,**kwargs)

def overview(d,out,previews):
 src=d.root/'R0/F1_example_with_text.json'
 # Panel (b) shows one real response; its annotated record is shipped, not recomputed.
 if not src.is_file():src=Path(__file__).resolve().parents[2]/'reference_results/R0/F1_example_with_text.json'
 e=json.loads(src.read_text(encoding='utf-8'))
 a=e['capture_analysis']; blocks=json.loads(e['response'])
 assert len(blocks)==74
 fig=plt.figure(figsize=(5.5,2.35))
 left=fig.add_axes([.015,.08,.36,.86]);left.set_axis_off()
 left.text(0,1,'(a) Block replacement',va='top',fontsize=8)
 left.text(.0,.84,'Cleaned targets',fontsize=7.5)
 left.text(.0,.48,'OBR targets',fontsize=7.5)
 for j in range(4):
  x=.045+j*.22
  for y in [.63,.27]:
   replaced=(y<.5 and j in [1,3]);color=COL['OBR'] if replaced else COL['cleaned']
   left.add_patch(Rectangle((x,y),.17,.13,facecolor=color,edgecolor='none'))
   left.text(x+.085,y+.065,str(j+1),ha='center',va='center',color='white',fontsize=7.5)
  left.annotate('',xy=(x+.085,.43),xytext=(x+.085,.59),arrowprops={'arrowstyle':'->','lw':.7,'color':'.45'})
 left.text(.0,.17,'Same input and block count',fontsize=7)
 for y,role,label in [(.07,'cleaned','Unchanged block'),(-.01,'OBR','Replaced block')]:
  left.add_patch(Rectangle((.01,y),.035,.04,facecolor=COL[role],edgecolor='none'))
  left.text(.065,y+.019,label,va='center',fontsize=7)
 ax=fig.add_axes([.49,.46,.485,.44]);style(ax)
 cov=np.array(a['gold_pairs_hit']); x=np.arange(1,len(cov)+1)
 ax.step(x,cov,where='post',color=COL['raw'])
 first=np.asarray(a['first_occurrence'],bool)
 ax.scatter(x[~first],np.full(np.sum(~first),-2),marker='|',s=28,color=COL['raw'],linewidths=.9)
 ax.axvline(44,color='.45',linestyle=':',lw=.7);ax.axvline(72,color=COL['OBR'],linestyle=':',lw=.8)
 ax.scatter([74],[cov[-1]],marker='o',s=12,color=COL['raw'])
 ax.set_xlim(1,75);ax.set_ylim(-4,53);ax.set_xticks([1,25,44,57,72]);ax.set_yticks([0,20,40])
 ax.set_xlabel('Complete block',labelpad=2);ax.set_ylabel('Reference pairs\ncovered',labelpad=2)
 ax.set_title('(b) One raw-model response',loc='left',pad=4)
 ax.annotate('First reuse',xy=(44,0),xytext=(23,11),fontsize=7,arrowprops={'arrowstyle':'-','lw':.5})
 ax.annotate('Capture confirmed',xy=(72,43),xytext=(36,49),fontsize=7,ha='center',arrowprops={'arrowstyle':'-','lw':.5})
 ax.annotate('Normal EOS',xy=(74,43),xytext=(57,27),fontsize=7,arrowprops={'arrowstyle':'-','lw':.5})
 # Faithful English translations of source, target and relation labels in blocks 67--72.
 inset=fig.add_axes([.43,.015,.56,.25]);inset.set_axis_off()
 inset.text(0,.91,'Two-relation motif, blocks 67–72',fontsize=7.5)
 inset.text(0,.64,'Source: Institutes of Justinian',fontsize=7)
 inset.text(0,.35,'composition of legal system  →  legal system',fontsize=7)
 inset.text(0,.06,'object of legal norms  →  legal norms',fontsize=7)
 save(fig,'F1_overview',out,previews)

def risks(d,pair):
 q=d.csv(f'R0/{pair}/p0d_episode_hazard/P0D_EPISODE_SUMMARY.csv')
 q=q[(q.episode_variant=='lineage')&(q.capture_kind=='primary')]
 z=q.pivot(index='episode_ordinal',columns='model',values='episodes_started')
 good=z.index[(z['M0']>=50)&(z['M1']>=50)]
 return [q[(q.model==m)&q.episode_ordinal.isin(good)].sort_values('episode_ordinal') for m in ['M0','M1']]

def dynamics(d,out,previews):
 fig=plt.figure(figsize=(5.5,2.8))
 a=fig.add_axes([.08,.565,.37,.325]);b=fig.add_axes([.60,.565,.37,.325]);c=fig.add_axes([.39,.13,.58,.25])
 for ax in (a,b,c):style(ax)
 a.set_title('(a) Reuse exposure and capture',loc='left')
 pts=[('qwen3-4b-notrain','Untuned','untuned','o',(-6,7)),(C,'Cleaned','cleaned','o',(-35,4)),(K,'Raw','raw','o',(-15,-12)),(O,'OBR 15%','OBR','o',(5,6)),('qwen3-4b-generic-noise-s42','Generic label\nnoise','control','s',(4,-7)),('qwen3-4b-isc-a-s42','Candidate\ninput edit','control','^',(-46,-10))]
 for tag,label,role,m,offset in pts:
  r=d.one('input/e1_metrics.csv',tag=tag);xx=r.episodes_per_response; yy=100*r.per_episode_capture_hazard
  a.scatter(xx,yy,color=COL[role],marker=m,s=17)
  a.annotate(label,(xx,yy),xytext=offset,textcoords='offset points',fontsize=7)
 a.set(xlim=(0,7.3),ylim=(-.2,5.7),xlabel='Episodes per response',ylabel='Captures per episode (%)')
 a.set_xticks([0,2,4,6]);a.set_yticks([0,2,4])
 b.set_title('(b) Capture by episode ordinal',loc='left')
 cc,kk=risks(d,'4b_CK_s42');_,oo=risks(d,'4b_CO15_s42')
 for x,label,role in [(cc,'Cleaned','cleaned'),(kk,'Raw','raw'),(oo,'OBR 15%','OBR')]:
  b.plot(x.episode_ordinal,100*x.capture_hazard_given_episode,color=COL[role],label=label)
 b.set(xlabel='Episode ordinal',ylabel='Capture probability (%)');b.set_xticks([1,8,16,24]);b.set_ylim(0,12)
 b.legend(loc='upper left',ncol=1,handlelength=1.4,labelspacing=.15,borderaxespad=.1)
 fig.text(.08,.397,'(c) Standardized components',fontsize=8)
 pairs=[('1p7b_CK_s42','Raw, 1.7B, seed 42'),('4b_CK_s42','Raw, 4B, seed 42'),('4b_CK_s123','Raw, 4B, seed 123'),('8b_CK_s42','Raw, 8B, seed 42'),('4b_CO15_s42','OBR 15%, 4B, seed 42'),('4b_CO15_s123','OBR 15%, 4B, seed 123')]
 for j,(p,l) in enumerate(pairs):
  for metric,marker,shift in [('exposure_component','o',-.13),('propensity_component','s',.13)]:
   r=d.one('R0/R0_pair_metrics.csv',pair=p,metric=metric)
   color=COL['OBR'] if 'CO15' in p else COL['raw']
   c.errorbar(r.diff_or_value,j+shift,xerr=[[r.diff_or_value-r.ci_low],[r.ci_high-r.diff_or_value]],color=color,marker=marker,linestyle='none',mfc='white' if marker=='o' else color,ms=3.3,capsize=1.5,elinewidth=.8)
 c.set_yticks(np.arange(6));c.set_yticklabels([l for p,l in pairs]);c.invert_yaxis();c.set_ylim(5.5,-.7)
 c.axvline(0,color='.6',lw=.6);c.set_xlim(-.002,.225);c.set_xticks([0,.05,.10,.15,.20]);c.set_xlabel('Contribution to capture incidence',labelpad=2)
 handles=[Line2D([0],[0],marker='o',color='.3',mfc='white',ls='',label='Exposure'),Line2D([0],[0],marker='s',color='.3',ls='',label='Persistence')]
 fig.legend(handles=handles,loc='center right',bbox_to_anchor=(.99,.405),ncol=2,handletextpad=.3,columnspacing=.8)
 save(fig,'F2_dynamics',out,previews)

def prefix(d,out,previews):
 fig,axs=plt.subplots(1,3,figsize=(5.5,1.9));fig.subplots_adjust(left=.12,right=.97,bottom=.32,top=.89,wspace=.70)
 roles=['natural_stop','early_remaining','pre_first_reuse']; labs=['Natural\nstop','Early','Before\nfirst reuse']
 series=[(K,'Raw, seed 42','raw','o',-.18),(K.replace('s42','s123'),'Raw, seed 123','raw','s',-.06),(O,'OBR 15%, seed 42','OBR','o',.06),(O.replace('s42','s123'),'OBR 15%, seed 123','OBR','s',.18)]
 for i,(metric,mult,ylabel) in enumerate([('next_event_close_rate',100,'Next-event close\nfrequency difference (pp)'),('margin_first_policy',1,'First-token stop margin\ndifference (nats)')]):
  ax=axs[i];style(ax);ax.axhline(0,color='.6',lw=.6)
  for tag,label,role,m,offset in series:
   rows=[d.one('R1/R1_paired_effects_4B.csv',metric=metric,m1_tag=tag,split='test',source='all',role=r) for r in roles]
   ys=np.array([r['diff'] for r in rows])*mult;lo=np.array([r.ci_low for r in rows])*mult;hi=np.array([r.ci_high for r in rows])*mult
   err(ax,np.arange(3)+offset,ys,lo,hi,fmt=m,color=COL[role],mfc='white' if m=='s' else COL[role],ms=3)
  ax.set_xticks(range(3));ax.set_xticklabels(labs);ax.set_ylabel(ylabel,labelpad=2);ax.set_title(f'({chr(97+i)}) 4B',loc='left');ax.set_xlim(-.35,2.35)
 ax=axs[2];style(ax);ax.axhline(0,color='.6',lw=.6)
 rows=[d.one('R1/R1_paired_effects_8B.csv',metric='next_event_close_rate',m1_tag=K8,split='test',source='all',role=r) for r in roles]
 err(ax,np.arange(3),np.array([r['diff'] for r in rows])*100,np.array([r.ci_low for r in rows])*100,np.array([r.ci_high for r in rows])*100,fmt='o',color=COL['raw'])
 ax.set_xticks(range(3));ax.set_xticklabels(labs);ax.set_ylabel('Next-event close\nfrequency difference (pp)',labelpad=2);ax.set_title('(c) 8B',loc='left');ax.set_xlim(-.35,2.35)
 for ax in [axs[0],axs[2]]:ax.set_ylim(-44,5);ax.set_yticks([-40,-20,0])
 handles=[Line2D([0],[0],marker=m,color=COL[role],mfc='white' if m=='s' else COL[role],ls='',label=label) for _,label,role,m,_ in series]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.005),ncol=2,handletextpad=.3,columnspacing=1.8,labelspacing=.25)
 save(fig,'F3_termination',out,previews)

def interventions(d,out,previews):
 fig,axs=plt.subplots(1,2,figsize=(5.5,1.8));fig.subplots_adjust(left=.11,right=.985,bottom=.36,top=.89,wspace=.4)
 series=[('4B',C,'4B cleaned reverse','cleaned','-','o'),('4B',K,'4B raw restore','raw','-','o'),('4B',O,'4B OBR 15% transfer','OBR','-','o'),('8B',C8,'8B cleaned reverse','cleaned','--','s'),('8B',K8,'8B raw restore','raw','--','s')]
 for i,(metric,mult,title,ylabel) in enumerate([('next_event_close_rate',100,'(a) Closing decision','Next-event close\nfrequency change (pp)'),('margin_first_policy',1,'(b) Stop margin','First-token stop margin\nchange (nats)')]):
  ax=axs[i];style(ax);ax.axhline(0,color='.6',lw=.6)
  for j,(scale,tag,label,role,ls,m) in enumerate(series):
   path=f'R2/R2_short_effects_{scale}.csv'
   rows=[d.one(path,model_tag=tag,metric=metric,condition=c,comparison='vs_baseline') for c in ['direction_a0.5','direction_a1','direction_a2']]
   x=np.array([0,.5,1,2]);y=np.array([0]+[r['diff'] for r in rows])*mult
   ax.plot(x,y,color=COL[role],ls=ls,marker=m,ms=3,label=label,mfc='white' if scale=='8B' else COL[role])
   lo=np.array([r.ci_low for r in rows])*mult;hi=np.array([r.ci_high for r in rows])*mult
   err(ax,x[1:],y[1:],lo,hi,fmt='none',ecolor=COL[role],alpha=.55)
   rr=[d.one(path,model_tag=tag,metric=metric,condition=f'random{k}_a1',comparison='vs_baseline')['diff']*mult for k in range(8)]
   ax.scatter(1+np.linspace(-.055,.055,8)+(j-2)*.012,rr,color=COL['random'],s=6,marker='.',zorder=5)
  ax.set_xticks([0,.5,1,2]);ax.set_xlabel(r'Pulse strength $\alpha$',labelpad=2);ax.set_ylabel(ylabel,labelpad=2);ax.set_title(title,loc='left');ax.set_xlim(-.04,2.06)
 handles=[Line2D([0],[0],color=COL[role],ls=ls,marker=m,mfc='white' if sc=='8B' else COL[role],label=label) for sc,tag,label,role,ls,m in series]
 handles.append(Line2D([0],[0],color=COL['random'],marker='.',ls='',label='Equal-norm random directions'))
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.002),ncol=3,handlelength=1.2,columnspacing=.7,handletextpad=.4,labelspacing=.25)
 save(fig,'F4_intervention',out,previews)

def remaining_curves(d,out,previews):
 pairs=[('1p7b_CK_s42','1.7B raw vs. cleaned','Raw','raw'),('4b_CK_s123','4B raw vs. cleaned, seed 123','Raw','raw'),('8b_CK_s42','8B raw vs. cleaned','Raw','raw'),('4b_CO15_s123','4B OBR 15% vs. cleaned, seed 123','OBR 15%','OBR'),('4b_notrainC_s42','4B cleaned vs. untuned','Cleaned','cleaned'),('4b_CGN_s42','4B generic label noise vs. cleaned','Generic label noise','control'),('4b_CISCA_s42','4B candidate input edit vs. cleaned','Candidate input edit','control')]
 for k,items in enumerate([pairs[:4],pairs[4:]]):
  n=len(items);fig,axs=plt.subplots(n,2,figsize=(5.5,1.38*n+.2));fig.subplots_adjust(left=.11,right=.985,bottom=.08,top=.94,hspace=.7,wspace=.4)
  for i,(pair,title,tlabel,role) in enumerate(items):
   rr=risks(d,pair);base='Untuned' if 'notrain' in pair else 'Cleaned';bc=COL['untuned'] if 'notrain' in pair else COL['cleaned']
   for j,(metric,label) in enumerate([('capture_hazard_given_episode','Capture probability (%)'),('gap_stop_hazard','Gap stop probability (%)')]):
    ax=axs[i,j];style(ax)
    for z,lab,color in zip(rr,[base,tlabel],[bc,COL[role]]):ax.plot(z.episode_ordinal,100*z[metric],label=lab,color=color)
    ax.set_xlabel('Episode ordinal',labelpad=1);ax.set_ylabel(label,labelpad=2);ax.set_title(f'({chr(97+2*i+j)}) '+title,loc='left',fontsize=7.5)
    ax.legend(fontsize=7,loc='best',handlelength=1.2,borderaxespad=.2,labelspacing=.2,frameon=True,facecolor='white',edgecolor='none',framealpha=.96)
  save(fig,f'A{1+k}_process_curves',out,previews)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,default=None,help='default: <results>/figures');p.add_argument('--metrics',type=Path,default=None,help='e1_metrics.csv when <results>/input/ is absent');p.add_argument('--previews',type=Path);args=p.parse_args()
 if args.output is None:args.output=args.results/'figures'
 if args.metrics is None and os.environ.get('EVAL_ROOT'):args.metrics=Path(os.environ['EVAL_ROOT'])/'e1_metrics.csv'
 if not args.results.is_dir():p.error('--results must be the extracted result archive root')
 args.output.mkdir(parents=True,exist_ok=True)
 if args.previews:args.previews.mkdir(parents=True,exist_ok=True)
 d=Data(args.results,args.metrics)
 for fn in [overview,dynamics,prefix,interventions,remaining_curves]:fn(d,args.output,args.previews)
if __name__=='__main__':main()
