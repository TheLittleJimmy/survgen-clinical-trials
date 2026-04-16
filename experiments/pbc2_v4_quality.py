#!/usr/bin/env python3
"""
Comprehensive control group generation quality analysis for V4 models.
Trains V4_joint with Weibull on PBC2 placebo group with tuned hyperparameters,
generates extensive quality visualizations from multiple perspectives.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))
import data_processing, surv_hivae
from src import HIVAE_inputDropout

DATA_DIR = ROOT / "dataset" / "pbc2"
BASE_OUT = ROOT / "experiments" / "pbc2_v4_quality"
for d_name in ["01_marginals", "02_joint_structure", "03_survival",
               "04_longitudinal", "05_multi_seed", "06_summary"]:
    (BASE_OUT / d_name).mkdir(parents=True, exist_ok=True)

LONG_NAMES = ["ascites","hepatomegaly","spiders","edema","serBilir","albumin",
              "alkaline","SGOT","platelets","prothrombin","histologic"]
LONG_CONT = ["serBilir","albumin","alkaline","SGOT","platelets","prothrombin"]
COLS = ["time","censor","drug","sex","ascites","hepatomegaly","spiders",
        "edema","histologic","albumin","alkaline","SGOT","platelets","prothrombin","age"]
CONT_COLS = ["albumin","alkaline","SGOT","platelets","prothrombin","age"]
CAT_COLS = ["sex","ascites","hepatomegaly","spiders","edema","histologic"]

def load_ctrl():
    pbc2 = pd.read_csv(DATA_DIR / "pbc2_id.csv")
    pm = pbc2['drug'] == 'placebo'; pids = pbc2.index[pm].tolist()
    sm={'female':0,'male':1}; ym={'No':0,'Yes':1}
    em={'No edema':0,'edema no diuretics':1,'untreated or successfully treated':1,'edema despite diuretics':2}
    c = pbc2.loc[pm].copy()
    c['drug_enc']=1; c['sex_enc']=c['sex'].map(sm); c['ascites_enc']=c['ascites'].map(ym)
    c['hepatomegaly_enc']=c['hepatomegaly'].map(ym); c['spiders_enc']=c['spiders'].map(ym)
    c['edema_enc']=c['edema'].map(em); c['censor']=c['status2'].astype(int); c['time']=c['years']
    oc=['time','censor','drug_enc','sex_enc','ascites_enc','hepatomegaly_enc',
        'spiders_enc','edema_enc','histologic','albumin','alkaline','SGOT','platelets','prothrombin','age']
    dc=c[oc].copy()
    for col in dc.columns:
        if dc[col].isna().any(): dc[col]=dc[col].fillna(dc[col].median())
    dc.to_csv(DATA_DIR/"data_v4_control.csv", index=False, header=False)
    lf=pd.read_csv(DATA_DIR/"longitudinal.csv")
    pm2={o:n for n,o in enumerate(pids)}
    lc=lf[lf['patient_id'].isin(pids)].copy()
    lc['patient_id']=lc['patient_id'].map(pm2)
    lc=lc.sort_values(['patient_id','visit_time']).reset_index(drop=True)
    t=pbc2.loc[~pm].copy()
    t['drug_enc']=0; t['sex_enc']=t['sex'].map(sm); t['ascites_enc']=t['ascites'].map(ym)
    t['hepatomegaly_enc']=t['hepatomegaly'].map(ym); t['spiders_enc']=t['spiders'].map(ym)
    t['edema_enc']=t['edema'].map(em); t['censor']=t['status2'].astype(int); t['time']=t['years']
    dt=t[oc].copy()
    for col in dt.columns:
        if dt[col].isna().any(): dt[col]=dt[col].fillna(dt[col].median())
    return dc, lc, dt, len(pids)

def train_model(version, seed, n, ldf, epochs=500, z_dim=30, y_dim=20, s_dim=25, lr=3e-4, bs=40, dev='cpu'):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    df,ty,mi,tm,n2=data_processing.read_data(str(DATA_DIR/"data_v4_control.csv"),
        str(DATA_DIR/"data_types_v4.csv"),"Missing.csv",None,surv_type='surv_weibull')
    tn,vn,mk,lnp=data_processing.prepare_longitudinal_tensors(ldf,
        patient_id_col="patient_id",time_col="visit_time",value_col=LONG_NAMES,n_patients=n2)
    ld=(tn,vn,mk); bs2=min(bs,int(0.9*n2))
    m=HIVAE_inputDropout(df.shape[1],z_dim=z_dim,y_dim=y_dim,s_dim=s_dim,
        y_dim_partition=None,feat_types_dict=ty,intervals_surv_piecewise=None,
        n_layers_surv_piecewise=None,model_version=version,n_long_outcomes=len(LONG_NAMES))
    dt=torch.from_numpy(df.values)
    import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m,lt,lv=surv_hivae.train_HIVAE(m,dt,mi,tm,ty,bs2,lr,epochs,verbose=False,longitudinal_data=ld,device=dev)
    g=surv_hivae.generate_from_HIVAE(m,dt,mi,tm,ty,10,device=dev)
    # Multiple generation rounds for averaging
    gens = [g]
    for _ in range(2):
        gens.append(surv_hivae.generate_from_HIVAE(m,dt,mi,tm,ty,10,device=dev))
    rd=data_processing.discrete_variables_transformation(dt,ty)
    rd=data_processing.survival_variables_transformation(rd,ty)
    # Longitudinal
    device2 = next(m.parameters()).device
    with torch.no_grad():
        tg=torch.linspace(0,1,30)
        dd=dt.to(device2); md2=torch.multiply(mi,tm).to(device2)
        dl,ml=data_processing.next_batch(dd,ty,md2,n2,0)
        do=[d*ml[:,j].view(n2,1) for j,d in enumerate(dl)]
        xl,_=data_processing.batch_normalization(do,ty,ml)
        X=torch.cat(xl,dim=1)
        ld2=tuple(t.to(device2) for t in ld)
        ls=m._encode_longitudinal_summary(ld2)
        X=torch.cat([X,ls],dim=1)
        _,sa=m.encode(X,tau=1e-3)
        if version=='v4_seq':
            nsx=[xl[fi] for fi,f in enumerate(ty) if not f['type'].startswith('surv')]
            cx=m.baseline_summary_net(torch.cat(nsx,dim=1))
            mu,va,tr=m.generate_longitudinal_seq(sa,tg.to(device2),cx,n_samples=10)
        else:
            mu,va,tr=m.generate_longitudinal(sa,tg.to(device2),n_samples=10)
    return {"gens":gens,"real":rd,"loss":lt,"mu":mu.cpu(),"traj":tr.cpu(),"tg":tg,
            "lnp":lnp,"ldf":ldf,"types":ty,"n":n2,"model":m}

def main():
    dc,lc,dt,nc = load_ctrl()
    print(f"Control group: {nc} patients")

    # Train with improved hyperparams (larger latent, more capacity)
    print("\n=== Training V4_joint (improved hyperparams) ===")
    rj = train_model("v4_joint", 42, nc, lc, epochs=500, z_dim=30, y_dim=20, s_dim=25, lr=3e-4, bs=40)
    print(f"  Final loss: {rj['loss'][-1]:.2f}")

    print("\n=== Training V4_seq ===")
    rs = train_model("v4_seq", 42, nc, lc, epochs=500, z_dim=30, y_dim=20, s_dim=25, lr=3e-4, bs=40)
    print(f"  Final loss: {rs['loss'][-1]:.2f}")

    for vname, res in [("v4_joint", rj), ("v4_seq", rs)]:
        label = vname.upper().replace("_"," ")
        # Pool multiple generation rounds
        all_syn = []
        for g in res["gens"]:
            for i in range(min(g.shape[0], 10)):
                syn_df = pd.DataFrame(g[i].numpy(), columns=COLS)
                all_syn.append(syn_df)
        syn_pool = pd.concat(all_syn, ignore_index=True)
        # Single best sample
        syn1 = pd.DataFrame(res["gens"][0][0].numpy(), columns=COLS)
        real_df = pd.DataFrame(res["real"].numpy(), columns=COLS)

        # ================================================================
        # 01: MARGINAL DISTRIBUTIONS
        # ================================================================
        # Continuous features
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        for i, c in enumerate(CONT_COLS):
            ax = axes[i]
            rv = real_df[c].values; sv = syn1[c].values
            bins = np.linspace(min(rv.min(), sv.min()), max(rv.max(), sv.max()), 30)
            ax.hist(rv, bins=bins, alpha=0.5, density=True, color='#2166ac', label='Real', edgecolor='white')
            ax.hist(sv, bins=bins, alpha=0.5, density=True, color='#d6604d', label='Synthetic', edgecolor='white')
            ks, p = stats.ks_2samp(rv, sv)
            ax.set_title(f"{c}\nKS={ks:.3f}, p={p:.3f}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=9); ax.set_ylabel("Density")
        fig.suptitle(f"{label}: Continuous Feature Marginals (Control Group)", fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"01_marginals"/f"{vname}_continuous.png", dpi=200); plt.close()
        print(f"  Saved: 01_marginals/{vname}_continuous.png")

        # Categorical features
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()
        for i, c in enumerate(CAT_COLS):
            ax = axes[i]
            rc = real_df[c].value_counts(normalize=True).sort_index()
            sc = syn1[c].value_counts(normalize=True).sort_index()
            all_c = sorted(set(rc.index)|set(sc.index))
            x = np.arange(len(all_c)); w = 0.35
            ax.bar(x-w/2, [rc.get(v,0) for v in all_c], w, color='#2166ac', alpha=0.7, label='Real')
            ax.bar(x+w/2, [sc.get(v,0) for v in all_c], w, color='#d6604d', alpha=0.7, label='Syn')
            ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in all_c])
            ax.set_title(c, fontsize=11, fontweight='bold'); ax.legend(fontsize=8)
        fig.suptitle(f"{label}: Categorical Distributions (Control)", fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"01_marginals"/f"{vname}_categorical.png", dpi=200); plt.close()
        print(f"  Saved: 01_marginals/{vname}_categorical.png")

        # ================================================================
        # 02: JOINT / CORRELATION STRUCTURE
        # ================================================================
        num_cols = [c for c in COLS if c not in ['drug','sex','ascites','hepatomegaly','spiders','censor']]
        rc = real_df[num_cols].corr(); sc = syn1[num_cols].corr()
        fig, (a1,a2,a3) = plt.subplots(1,3,figsize=(22,7))
        for ax,mat,ttl in [(a1,rc,'Real'),(a2,sc,'Synthetic'),(a3,rc-sc,'Difference')]:
            vmin,vmax = (-1,1) if 'Diff' not in ttl else (-(rc-sc).abs().max().max(), (rc-sc).abs().max().max())
            im=ax.imshow(mat.values,cmap='RdBu_r',vmin=vmin,vmax=vmax)
            ax.set_title(ttl,fontweight='bold',fontsize=12)
            ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols,rotation=45,ha='right',fontsize=8)
            ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols,fontsize=8)
            plt.colorbar(im,ax=ax,fraction=0.046)
        fig.suptitle(f"{label}: Correlation Matrix Comparison",fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"02_joint_structure"/f"{vname}_correlation.png",dpi=200); plt.close()
        print(f"  Saved: 02_joint_structure/{vname}_correlation.png")

        # Pairwise scatter: key pairs
        pairs = [("albumin","time"),("age","time"),("alkaline","SGOT"),("prothrombin","time")]
        fig, axes = plt.subplots(2,2,figsize=(12,12))
        for idx,(cx,cy) in enumerate(pairs):
            ax = axes.ravel()[idx]
            ax.scatter(real_df[cx],real_df[cy],alpha=0.4,s=15,color='#2166ac',label='Real')
            ax.scatter(syn1[cx],syn1[cy],alpha=0.4,s=15,color='#d6604d',label='Syn')
            ax.set_xlabel(cx); ax.set_ylabel(cy); ax.legend(fontsize=8)
            ax.set_title(f"{cx} vs {cy}",fontweight='bold')
        fig.suptitle(f"{label}: Pairwise Scatter Plots",fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"02_joint_structure"/f"{vname}_scatter.png",dpi=200); plt.close()
        print(f"  Saved: 02_joint_structure/{vname}_scatter.png")

        # ================================================================
        # 03: SURVIVAL ANALYSIS
        # ================================================================
        rt = real_df["time"].values; re = real_df["censor"].values
        st = syn1["time"].values.clip(0); se = (syn1["censor"].values>0.5).astype(float)
        xmax = rt.max()*1.05

        # KM curves
        fig, ax = plt.subplots(figsize=(10,7))
        kmf = KaplanMeierFitter()
        kmf.fit(rt,re,label='Real Control'); kmf.plot_survival_function(ax=ax,color='#2166ac',linewidth=2.5)
        kmf.fit(st.clip(0,xmax),se,label='Synthetic'); kmf.plot_survival_function(ax=ax,color='#d6604d',linewidth=2,linestyle='--')
        lr = logrank_test(rt,st.clip(0,xmax),re,se)
        ax.set_xlim(0,xmax); ax.set_xlabel("Time (years)",fontsize=12); ax.set_ylabel("Survival Probability",fontsize=12)
        ax.set_title(f"{label}: Kaplan-Meier (Control)\nLog-rank p={lr.p_value:.4f}",fontsize=14,fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout(); plt.savefig(BASE_OUT/"03_survival"/f"{vname}_km.png",dpi=200); plt.close()
        print(f"  Saved: 03_survival/{vname}_km.png")

        # Time distribution + QQ + event rate
        fig = plt.figure(figsize=(18,5)); gs = gridspec.GridSpec(1,3,figure=fig)
        ax1=fig.add_subplot(gs[0,0])
        bins=np.linspace(0,max(rt.max(),st.max()),30)
        ax1.hist(rt,bins=bins,alpha=0.5,density=True,color='#2166ac',label='Real')
        ax1.hist(st.clip(0),bins=bins,alpha=0.5,density=True,color='#d6604d',label='Syn')
        ks,p=stats.ks_2samp(rt,st); ax1.set_title(f"Survival Time Dist\nKS={ks:.3f} p={p:.3f}",fontweight='bold')
        ax1.legend()
        ax2=fig.add_subplot(gs[0,1])
        nq=min(len(rt),len(st)); q=np.linspace(0,1,nq)
        rq=np.quantile(rt,q); sq=np.quantile(st,q)
        ax2.scatter(rq,sq,s=10,alpha=0.5,color='purple')
        lm=[0,max(rq.max(),sq.max())*1.05]; ax2.plot(lm,lm,'k--')
        ax2.set_xlabel("Real Quantiles"); ax2.set_ylabel("Syn Quantiles"); ax2.set_title("QQ Plot",fontweight='bold')
        ax3=fig.add_subplot(gs[0,2])
        x=np.arange(2); w=0.5
        ax3.bar(x,[re.mean(),se.mean()],w,color=['#2166ac','#d6604d'],alpha=0.7)
        ax3.set_xticks(x); ax3.set_xticklabels(['Real','Syn'])
        ax3.set_title(f"Event Rate\nReal={re.mean():.3f} Syn={se.mean():.3f}",fontweight='bold')
        ax3.set_ylabel("Event Rate")
        fig.suptitle(f"{label}: Survival Diagnostics",fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"03_survival"/f"{vname}_diagnostics.png",dpi=200); plt.close()
        print(f"  Saved: 03_survival/{vname}_diagnostics.png")

        # Replacement KM: Treatment vs Real Ctrl vs Syn Ctrl
        fig, (a1,a2) = plt.subplots(1,2,figsize=(18,7))
        tt=dt['time'].values; te=dt['censor'].values; xm2=max(tt.max(),rt.max())*1.05
        kmf.fit(tt,te,label='Treatment'); kmf.plot_survival_function(ax=a1,color='#2166ac',linewidth=2.5)
        kmf.fit(rt,re,label='Real Control'); kmf.plot_survival_function(ax=a1,color='#b2182b',linewidth=2.5)
        lr1=logrank_test(tt,rt,te,re); a1.set_xlim(0,xm2)
        a1.set_title(f"Treat vs Real Ctrl\np={lr1.p_value:.4f}",fontweight='bold'); a1.legend()
        kmf.fit(tt,te,label='Treatment'); kmf.plot_survival_function(ax=a2,color='#2166ac',linewidth=2.5)
        kmf.fit(st.clip(0,xm2),se,label='Syn Control'); kmf.plot_survival_function(ax=a2,color='#d6604d',linewidth=2,linestyle='--')
        lr2=logrank_test(tt,st.clip(0,xm2),te,se); a2.set_xlim(0,xm2)
        a2.set_title(f"Treat vs Syn Ctrl\np={lr2.p_value:.4f}",fontweight='bold'); a2.legend()
        for ax in [a1,a2]: ax.set_xlabel("Time (years)"); ax.set_ylabel("Survival Prob")
        fig.suptitle(f"{label}: Replacement Analysis",fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.savefig(BASE_OUT/"03_survival"/f"{vname}_replacement.png",dpi=200); plt.close()
        print(f"  Saved: 03_survival/{vname}_replacement.png")

        # ================================================================
        # 04: LONGITUDINAL TRAJECTORIES
        # ================================================================
        lnp = res["lnp"]
        for oname in LONG_CONT:
            oidx = LONG_NAMES.index(oname)
            fig, ax = plt.subplots(figsize=(10,6))
            tmin,tmax_l = lnp["time_min"],lnp["time_max"]
            tgo = res["tg"]*(tmax_l-tmin)+tmin
            vm,vs = lnp["value_mean"][oidx],lnp["value_std"][oidx]
            mu_d = res["mu"][:,:,oidx].numpy()*vs+vm
            tr_d = res["traj"][:,:,:,oidx].numpy()*vs+vm
            mu_m = mu_d.mean(axis=0)
            tf = tr_d.reshape(-1,tr_d.shape[-1])
            q5,q95 = np.percentile(tf,5,axis=0), np.percentile(tf,95,axis=0)
            ax.scatter(lc["visit_time"].values, lc[oname].values,alpha=0.15,s=8,color='#2166ac',label='Real')
            ax.plot(tgo.numpy(),mu_m,color='#d6604d',linewidth=2,label='Syn mean')
            ax.fill_between(tgo.numpy(),q5,q95,color='#d6604d',alpha=0.2,label='90% CI')
            ax.set_xlabel("Time (years)"); ax.set_ylabel(oname)
            ax.set_title(f"{label}: {oname} Trajectories (Control)",fontweight='bold')
            ax.legend()
            plt.tight_layout(); plt.savefig(BASE_OUT/"04_longitudinal"/f"{vname}_{oname}.png",dpi=200); plt.close()
        print(f"  Saved: 04_longitudinal/{vname}_*.png (6 files)")

    # ================================================================
    # 05: MULTI-SEED COMPARISON (3 seeds, lighter)
    # ================================================================
    print("\n=== Multi-seed stability (3 seeds) ===")
    seed_results = {}
    for vname in ["v4_joint", "v4_seq"]:
        sr = []
        for seed in [1, 42, 123]:
            print(f"  {vname} seed={seed}...")
            r = train_model(vname, seed, nc, lc, epochs=300, z_dim=30, y_dim=20, s_dim=25, lr=3e-4, bs=40)
            rdf = pd.DataFrame(r["real"].numpy(), columns=COLS)
            sdf = pd.DataFrame(r["gens"][0][0].numpy(), columns=COLS)
            rt2=rdf["time"].values; re2=rdf["censor"].values
            st2=sdf["time"].values.clip(0); se2=(sdf["censor"].values>0.5).astype(float)
            lr_c = logrank_test(rt2,st2.clip(0,rt2.max()),re2,se2)
            lr_r = logrank_test(dt['time'].values,st2.clip(0,dt['time'].max()),dt['censor'].values,se2)
            ks,_=stats.ks_2samp(rt2,st2)
            sr.append({"seed":seed,"p_ctrl_syn":lr_c.p_value,"p_treat_syn":lr_r.p_value,
                       "ks":ks,"event_rate":se2.mean()})
            print(f"    p_ctrl={lr_c.p_value:.4f} p_treat={lr_r.p_value:.4f} ks={ks:.3f} er={se2.mean():.3f}")
        seed_results[vname] = sr

    with open(BASE_OUT/"05_multi_seed"/"seed_results.json","w") as f:
        json.dump(seed_results,f,indent=2,default=lambda x:float(x) if hasattr(x,'item') else x)

    # Seed stability plot
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    for vn,clr in [("v4_joint","#d6604d"),("v4_seq","#4daf4a")]:
        sr=seed_results[vn]; seeds=[r["seed"] for r in sr]
        axes[0].plot(seeds,[r["p_treat_syn"] for r in sr],'o-',color=clr,label=vn,markersize=8)
        axes[1].plot(seeds,[r["ks"] for r in sr],'o-',color=clr,label=vn,markersize=8)
        axes[2].plot(seeds,[r["event_rate"] for r in sr],'o-',color=clr,label=vn,markersize=8)
    axes[0].axhline(0.05,color='red',linestyle='--'); axes[0].set_ylabel("p-value"); axes[0].set_title("Treat vs Syn Ctrl",fontweight='bold')
    real_er = dc['censor'].mean()
    axes[2].axhline(real_er,color='black',linestyle='--',label=f'Real={real_er:.3f}')
    for ax in axes: ax.set_xlabel("Seed"); ax.legend(fontsize=9)
    axes[1].set_ylabel("KS stat"); axes[1].set_title("Survival Time KS",fontweight='bold')
    axes[2].set_ylabel("Event Rate"); axes[2].set_title("Event Rate",fontweight='bold')
    fig.suptitle("Multi-Seed Stability",fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(BASE_OUT/"05_multi_seed"/"stability.png",dpi=200); plt.close()
    print(f"  Saved: 05_multi_seed/stability.png")

    # ================================================================
    # 06: SUMMARY DASHBOARD
    # ================================================================
    fig = plt.figure(figsize=(20,12))
    gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.35,wspace=0.3)
    # Panel 1: Loss curves
    a1 = fig.add_subplot(gs[0,0])
    a1.plot(rj["loss"],label="V4_joint",color="#d6604d"); a1.plot(rs["loss"],label="V4_seq",color="#4daf4a")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Neg ELBO"); a1.set_title("Training Loss",fontweight='bold')
    a1.legend()
    # Panel 2: KM comparison
    a2 = fig.add_subplot(gs[0,1])
    kmf=KaplanMeierFitter()
    rdf_j=pd.DataFrame(rj["real"].numpy(),columns=COLS)
    kmf.fit(rdf_j["time"],rdf_j["censor"],label="Real"); kmf.plot_survival_function(ax=a2,color='black',linewidth=2)
    for vn,g,clr,ls in [("V4j",rj,"#d6604d","--"),("V4s",rs,"#4daf4a","-.")]:
        sd=pd.DataFrame(g["gens"][0][0].numpy(),columns=COLS)
        kmf.fit(sd["time"].clip(0,rdf_j["time"].max()),
                (sd["censor"]>0.5).astype(float),label=vn)
        kmf.plot_survival_function(ax=a2,color=clr,linewidth=2,linestyle=ls)
    a2.set_xlim(0,rdf_j["time"].max()*1.05); a2.set_title("KM Comparison",fontweight='bold')
    a2.legend(fontsize=9)
    # Panel 3: Event rate
    a3 = fig.add_subplot(gs[0,2])
    sj=pd.DataFrame(rj["gens"][0][0].numpy(),columns=COLS)
    ss=pd.DataFrame(rs["gens"][0][0].numpy(),columns=COLS)
    ers=[rdf_j["censor"].mean(),(sj["censor"]>0.5).astype(float).mean(),(ss["censor"]>0.5).astype(float).mean()]
    a3.bar(["Real","V4_joint","V4_seq"],ers,color=['black','#d6604d','#4daf4a'],alpha=0.7)
    a3.set_ylabel("Event Rate"); a3.set_title("Event Rate Comparison",fontweight='bold')
    for i,v in enumerate(ers): a3.text(i,v+0.01,f"{v:.3f}",ha='center',fontsize=10)
    # Panel 4-6: Key marginals
    for pi,c in enumerate(["albumin","age","prothrombin"]):
        ax=fig.add_subplot(gs[1,pi])
        rv=rdf_j[c].values
        sv_j=sj[c].values; sv_s=ss[c].values
        bins=np.linspace(min(rv.min(),sv_j.min(),sv_s.min()),max(rv.max(),sv_j.max(),sv_s.max()),25)
        ax.hist(rv,bins=bins,alpha=0.4,density=True,color='black',label='Real')
        ax.hist(sv_j,bins=bins,alpha=0.4,density=True,color='#d6604d',label='V4j')
        ax.hist(sv_s,bins=bins,alpha=0.4,density=True,color='#4daf4a',label='V4s')
        ax.set_title(c,fontweight='bold'); ax.legend(fontsize=8)
    fig.suptitle("V4 Control Group Generation Quality Dashboard",fontsize=16,fontweight='bold')
    plt.savefig(BASE_OUT/"06_summary"/"dashboard.png",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  Saved: 06_summary/dashboard.png")

    # Count figures
    total = sum(len(list((BASE_OUT/d).glob("*.png"))) for d in os.listdir(BASE_OUT) if (BASE_OUT/d).is_dir())
    print(f"\n{'='*60}")
    print(f"Total figures: {total}")
    for d in sorted(os.listdir(BASE_OUT)):
        dp = BASE_OUT/d
        if dp.is_dir():
            n_f = len(list(dp.glob("*.png")))
            if n_f > 0: print(f"  {d}/ — {n_f} figures")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
