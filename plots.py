"""Render the seven manuscript figures, using their established visual style."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

def draw_all(frames, output):
    OUT=Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    COPY=OUT.parent/'figure_data'
    rd=lambda key:frames[key].copy()
    checks={}
    def save(fig, number, tight=False):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        outside=[]
        for text in fig.findobj(mpl.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            box=text.get_window_extent(renderer)
            if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
                outside.append(text.get_text())
        checks[f'figure{number}_export']={'canvas_extended_for_labels':bool(outside),'labels':outside}
        kwargs={'bbox_inches':'tight','pad_inches':.1} if tight or outside else {}
        for ext in ('png','pdf'):
            fig.savefig(OUT/f'Figure{number}.{ext}',dpi=600,**kwargs)
        plt.close(fig)
    mpl.rcParams.update({'font.family':'Arial','font.size':16,'axes.labelsize':22,
     'xtick.labelsize':16,'ytick.labelsize':16,'axes.linewidth':.9,
     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
     'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    RED='#F8766D';TEAL='#00BFC4'

    def grid_axes(fig,rect,fs=16,label_fs=22):
     ax=fig.add_axes(rect)
     ax.set_axisbelow(True)
     ax.grid(which='major',color='#999999',linewidth=.45)
     ax.grid(which='minor',color='#BFBFBF',linewidth=.35)
     ax.tick_params(axis='both',which='major',length=5.5,width=.9,labelsize=fs,pad=4)
     ax.tick_params(which='minor',length=0)
     for s in ax.spines.values():s.set_linewidth(.9);s.set_color('black')
     ax.xaxis.label.set_size(label_fs);ax.yaxis.label.set_size(label_fs)
     return ax

    def years(ax):
     ax.set_xticks(range(2017,2023));ax.xaxis.set_minor_locator(MultipleLocator(.5))
     ax.set_xlim(2016.75,2022.25);ax.set_xlabel('Year')

    def percent(ax,major,minor):
     lo,hi=ax.get_ylim()
     ax.yaxis.set_major_locator(FixedLocator(np.arange(np.ceil(lo/major)*major,hi+.0001,major)))
     ax.yaxis.set_minor_locator(MultipleLocator(minor))
     ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:'0' if np.isclose(x,0) else f'{x:g}%'))

    def legend(ax,labels,title,fs=17,title_fs=20,anchor=(1.08,.53)):
     leg=ax.legend(labels,title=title,loc='center left',bbox_to_anchor=anchor,
      frameon=False,fontsize=fs,title_fontsize=title_fs,handlelength=1.35,
      handletextpad=.65,labelspacing=.7,borderaxespad=0)
     leg._legend_box.align='left'
     return leg

    # Figure 1: original black line, grid, range, typography and aspect ratio.
    f=rd('figure1');f=f[f.year.between(2010,2023)]
    fig=plt.figure(figsize=(8,5))
    ax=grid_axes(fig,[.167,.179,.813,.789],fs=18,label_fs=24)
    line=ax.plot(f.year,f.records,color='black',linewidth=2)[0]
    ax.set_xlim(2009.36,2023.64);ax.set_ylim(-3900,83900)
    ax.set_xticks(range(2010,2024,2));ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_yticks(range(0,80001,20000));ax.yaxis.set_minor_locator(MultipleLocator(10000))
    ax.set_xlabel('Year');ax.set_ylabel('PwC Publications')
    np.testing.assert_allclose(line.get_ydata(),f.records)
    save(fig,1);checks['figure1']={'points':len(f),'partial_year':2023,'population':'arXiv-linked raw snapshot records with valid standardized identifiers; publication date year'}

    # Figure 2: citation coral, topic teal; original right-hand legend.
    a=rd('annual_results')
    fig=plt.figure(figsize=(8,5))
    ax=grid_axes(fig,[.128,.160,.572,.811])
    lc=ax.plot(a.year,a.citation_paper_share,color=RED,linewidth=1.8)[0]
    lt=ax.plot(a.year,a.topic_paper_share,color=TEAL,linewidth=1.8)[0]
    years(ax);ax.set_ylim(-1.25,26.25);percent(ax,5,2.5)
    ax.set_ylabel('Share of LIS publications identified')
    ax.yaxis.label.set_size(20)
    legend(ax,['Citation','Topic'],'Query Method',anchor=(1.09,.53))
    np.testing.assert_allclose(lc.get_ydata(),a.ml_citing_publications/a.lis_publications*100)
    np.testing.assert_allclose(lt.get_ydata(),a.topic_publications/a.lis_publications*100)
    save(fig,2)

    # Figure 3: keep the original two vertically stacked panels.
    fig=plt.figure(figsize=(6,6))
    ax=grid_axes(fig,[.12,.591,.59,.393],fs=11,label_fs=15)
    ax.plot(a.year,a.ml_links,color=RED,linewidth=1.25)
    ax.plot(a.year,a.ml_citing_publications,color=TEAL,linewidth=1.25)
    years(ax);ax.set_ylabel('Count');ax.set_ylim(0,3820)
    ax.yaxis.set_label_coords(-.124,.5);ax.tick_params(length=3.5)
    ax.set_yticks([0,1000,2000,3000]);ax.yaxis.set_minor_locator(MultipleLocator(500))
    legend(ax,['Citations','Citing Papers'],'Type',fs=11.5,title_fs=14,anchor=(1.08,.51))
    fig.text(.014,.965,'A',fontsize=18,weight='bold')
    ax=grid_axes(fig,[.116,.089,.571,.395],fs=11,label_fs=15)
    ax.plot(a.year,a.citation_paper_share,color=RED,linewidth=1.25)
    ax.plot(a.year,a.ml_reference_share,color=TEAL,linewidth=1.25)
    years(ax);ax.set_ylabel('Share');ax.set_ylim(-.75,20.25);percent(ax,5,2.5)
    ax.yaxis.set_label_coords(-.124,.5);ax.tick_params(length=3.5)
    legend(ax,['% Citing papers','% ML references'],'Type',fs=11.5,title_fs=14,anchor=(1.08,.51))
    fig.text(.014,.465,'B',fontsize=18,weight='bold')
    save(fig,3)

    # Figure 4: keep original task/method panels and first/second-level colours.
    c=rd('figure4_coverage');fig=plt.figure(figsize=(8,8))
    for k,(kind,rect,letter,letter_y) in enumerate([
     ('task',[.128,.588,.583,.395],'A',.958),
     ('method',[.128,.089,.583,.395],'B',.459)]):
     ax=grid_axes(fig,rect,fs=15.5,label_fs=22)
     for lev,col in [(1,RED),(2,TEAL)]:
      sub=c[(c.kind==kind)&(c.level==lev)].sort_values('year')
      line=ax.plot(sub.year,sub.coverage_pct,color=col,linewidth=1.8)[0]
      np.testing.assert_allclose(line.get_ydata(),sub.observed_catalog_entities/sub.catalog_entities*100)
     years(ax);ax.set_ylim(-5,105);percent(ax,25,12.5);ax.set_ylabel('Share')
     ax.yaxis.set_label_coords(-.14,.5)
     legend(ax,['First-Level','Second-level'],'Type',fs=16.5,anchor=(1.07,.51))
     fig.text(.014,letter_y,letter,fontsize=24,weight='bold')
    save(fig,4)

    # Figure 5: original overlay (not stacked, and not side-by-side); original order.
    order=['Geology','Mathematics','Philosophy','Economics','History','Psychology',
     'Sociology','Chemistry','Medicine','Biology','Physics','Engineering','Materials Science',
     'Business','Environmental Science','Political Science','Art','Computer Science','Geography','ML']
    f=rd('figure5_fields');ml=rd('citation_age').iloc[0]
    f=pd.concat([f,pd.DataFrame([{'field':'ML','mean':ml['mean'],'median':ml['median'],'count':ml['n']}])],ignore_index=True).set_index('field').loc[order]
    fig=plt.figure(figsize=(8,8/1.5))
    ax=grid_axes(fig,[.351,.149,.631,.824],fs=16,label_fs=21)
    y=np.arange(len(f))[::-1]
    bm=ax.barh(y,f['mean'],height=.9,color='#BEBEBE',edgecolor='none')
    bd=ax.barh(y,f['median'],height=.9,color='#D77476',edgecolor='none')
    ax.set_yticks(y,order);ax.set_ylim(-.55,19.55);ax.set_xlim(-.85,18.65)
    ax.set_xticks([0,5,10,15]);ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.set_xlabel('Citation Age (Years)');ax.set_ylabel('Field Being Cited in LIS Literature')
    np.testing.assert_allclose([v.get_width() for v in bm],f['mean'])
    np.testing.assert_allclose([v.get_width() for v in bd],f['median'])
    save(fig,5)

    # Figure 6: original Times New Roman, blue scatter/histograms/KDE, red LOWESS.
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':12})
    f=rd('figure6');x=f.ml_share.to_numpy();y=f.nonml_mean_age.to_numpy()
    assert len(f)==3248 and np.isfinite(x).all() and np.isfinite(y).all()
    fit=lowess(y,x,frac=2/3,it=3,return_sorted=True)
    pd.DataFrame({'ml_share':fit[:,0],'nonmatched_age_lowess':fit[:,1]}).to_csv(COPY/'figure6_lowess.csv',index=False)
    fig=plt.figure(figsize=(8,8*1278/1269))
    ax=fig.add_axes([.085,.082,.747,.751]);tx=fig.add_axes([.085,.854,.747,.134],sharex=ax)
    ty=fig.add_axes([.857,.082,.129,.751],sharey=ax)
    ax.scatter(x,y,s=36,color='#4C72B0',edgecolors='white',linewidths=.45)
    ax.plot(fit[:,0],fit[:,1],color='red',linewidth=1.6)
    ax.set_xlim(-.045,1.05);ax.set_ylim(-1.875,39.875)
    ax.set_xticks(np.arange(0,1.01,.2));ax.set_yticks(np.arange(0,36,5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0%}'))
    ax.set_xlabel('Share of ML in LIS citations',fontsize=16)
    ax.set_ylabel('Mean time Lag between LIS and its nonmatched references',fontsize=16)
    ax.tick_params(labelsize=12,length=0,pad=9);ax.grid(False)
    for axis in [ax,tx,ty]:
     for s in ['top','right']:axis.spines[s].set_visible(False)
     for s in ['left','bottom']:axis.spines[s].set_color('#CCCCCC');axis.spines[s].set_linewidth(1.1)
     axis.set_axisbelow(True)
    sns.histplot(x=x,ax=tx,kde=True,color='#4C72B0',alpha=.5,edgecolor='white',linewidth=.4,line_kws={'linewidth':1.5})
    sns.histplot(y=y,ax=ty,kde=True,color='#4C72B0',alpha=.5,edgecolor='white',linewidth=.4,line_kws={'linewidth':1.5})
    tx.set_xlabel('');tx.set_ylabel('');ty.set_xlabel('');ty.set_ylabel('')
    tx.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    ty.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    tx.set_yticks([]);ty.set_xticks([])
    tx.spines['left'].set_visible(False);ty.spines['bottom'].set_visible(False)
    tx.grid(axis='x',color='#CCCCCC',linewidth=.8);ty.grid(axis='y',color='#CCCCCC',linewidth=.8)
    save(fig,6)
    checks['figure6']={'n':len(f),'curve':'statsmodels LOWESS, frac=2/3, it=3 (original defaults); not a Pearson regression line','marginals':'seaborn histplot bins=auto, KDE=True','statistics':rd('figure6_statistics').to_dict('records')[0]}


    f=rd("figure7")
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':8.8,'axes.labelsize':9.8,
                        'xtick.labelsize':8.3,'ytick.labelsize':8.3,'axes.linewidth':.8,
                        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600})
    colors={'Over-represented':'#1F78A8','Near proportional':'#92989C','Under-represented':'#C85F3E'}
    full_labels={'Natural Language Processing':(2.7,1.92),'Computer Vision':(11,-1.30),
                 'Medical':(.90,1.20),'Graphs':(5.0,1.01),'Time Series':(5.4,-.25),
                 'Speech':(1.1,-1.85),'Reasoning':(.52,.62),'Translation':(3.0,-.60),
                 'Sentiment Classification':(.055,3.12),'Tokenization':(.012,4.88),
                 'Adversarial':(3.4,-.94),'Computer Code':(.32,-1.20)}
    def plot():
        stem="Figure7"
        fig,ax=plt.subplots(figsize=(7.2,4.9))
        xlim=(.00020,60)
        ylim=(-4.85,7.35)
        ax.axhspan(0,ylim[1],color='#EAF3F8',zorder=0)
        ax.axhspan(ylim[0],0,color='#FBF0EA',zorder=0)
        ax.axhline(0,color='#6C6C6C',lw=.9,ls=(0,(4,4)),zorder=1)
        ax.axhline(1,color='#A9C7D8',lw=.5,ls=':',zorder=1)
        ax.axhline(-1,color='#D9B5A5',lw=.5,ls=':',zorder=1)
        maximum=f.lis_share_pct.max()
        for key in ['Near proportional','Under-represented','Over-represented']:
            sub=f[f.direction==key]
            ax.scatter(sub.pwc_share_pct,sub.selectivity_log2,
                       s=16+np.sqrt(sub.lis_share_pct/maximum)*410,color=colors[key],
                       alpha=.76 if key!='Near proportional' else .42,
                       edgecolor='white',linewidth=.55,zorder=4 if key!='Near proportional' else 3)
        for task,xytext in full_labels.items():
            r=f[f.task==task].iloc[0]
            ax.annotate(task,xy=(r.pwc_share_pct,r.selectivity_log2),xytext=xytext,textcoords='data',
                        fontsize=8.1,color='#222222',
                        arrowprops={'arrowstyle':'-','color':'#6D6D6D','lw':.45,'alpha':.72,'shrinkA':2,'shrinkB':2},zorder=7)
        ax.set_xscale('log');ax.set_xlim(*xlim);ax.set_ylim(*ylim)
        ticks=[.0005,.001,.005,.01,.05,.1,.5,1,5,10,40]
        ax.set_xticks(ticks);ax.set_xticklabels([f'{v:g}' for v in ticks])
        ax.set_yticks(range(-4,8))
        ax.set_xlabel('PwC task share by task category or label, log scale (%)')
        ax.set_ylabel(r'Selectivity index, log$_2$(LIS citation share / PwC task share)')
        ax.grid(True,axis='x',which='major',color='#D8D8D8',lw=.55)
        ax.grid(True,axis='y',which='major',color='#E1E1E1',lw=.5);ax.set_axisbelow(True)
        for s in ['top','right']:ax.spines[s].set_visible(False)
        left=.00040
        ax.text(left,ylim[1]-.23,'Over-represented in LIS citations',color='#1F78A8',weight='bold',
                fontsize=11.2,ha='left',va='top')
        ax.text(left,ylim[0]+.23,'Under-represented in LIS citations',color='#B64F32',weight='bold',
                fontsize=11.2,ha='left',va='bottom')
        ax.text(42,.08,'Proportional representation',color='#565656',fontsize=7.5,ha='right',va='bottom')
        fig.tight_layout();fig.canvas.draw()
        within=(f.pwc_share_pct.between(*xlim)&f.selectivity_log2.between(*ylim))
        assert within.all()
        # Confirm rendering does not alter point coordinates and the label boxes fit the output canvas.
        offsets=np.vstack([c.get_offsets().data for c in ax.collections])
        assert len(offsets)==len(f)
        renderer=fig.canvas.get_renderer()
        outside=[]
        for t in ax.texts+[ax.xaxis.label,ax.yaxis.label]:
            bb=t.get_window_extent(renderer)
            if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:outside.append(t.get_text())
        for ext in ['png','pdf']:
            fig.savefig(OUT/f'{stem}.{ext}',bbox_inches='tight')
        result={'points_submitted':len(f),'points_with_centres_inside_axes':int(within.sum()),
                'xlim':xlim,'ylim':ylim,'text_outside_canvas_before_tight_export':outside}
        plt.close(fig);return result


    checks['figure7']=plot()
    (OUT.parent/'plot_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
