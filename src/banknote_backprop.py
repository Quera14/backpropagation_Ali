"""Studi kasus Banknote Authentication: Backpropagation manual (NumPy).

Materi pendamping belajar, bukan pengganti analisis mandiri yang diminta dosen.
DATA: UCI Banknote Authentication, https://archive.ics.uci.edu/dataset/267/banknote+authentication
Inti NN / gradien ditulis di sini, scikit-learn hanya untuk split dan baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen, Request
import argparse
import csv
import json
import os
import platform
import time

import numpy as np

COLS = ['variance','skewness','curtosis','entropy','class']
DATA_URLS = [
  'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
  'https://archive.ics.uci.edu/static/public/267/data_banknote_authentication.txt',
]


def load_banknote(file_path=None):
    """Unduh data asli dari UCI (online Colab) atau gunakan berkas lokal.
    Dilarang mengganti dengan data acak untuk laporan hasil Banknote.
    """
    import pandas as pd
    candidate = Path(file_path) if file_path else Path('data_banknote_authentication.txt')
    if not candidate.exists():
        errors = []
        for url in DATA_URLS:
            try:
                req=Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urlopen(req,timeout=20) as response:
                    data=response.read()
                if len(data)<40_000: raise ValueError('Berkas terlalu kecil untuk dataset UCI.')
                candidate.write_bytes(data)
                break
            except Exception as exc:
                errors.append(f'{url}: {exc}')
        if not candidate.exists():
            raise FileNotFoundError(
                'Dataset UCI belum ada. Unduh data_banknote_authentication.txt dari '
                'https://archive.ics.uci.edu/dataset/267/banknote+authentication; '
                'unggah ke Files Colab di /content atau letakkan di folder kerja. '\
                'Kesalahan koneksi: ' + '; '.join(errors)
            )
    df=pd.read_csv(candidate,header=None,names=COLS)
    if df.shape != (1372,5):
        raise ValueError(f'Dataset tidak cocok: diperoleh {df.shape}, seharusnya (1372,5).')
    if df.isna().any().any(): raise ValueError('Ditemukan nilai kosong; cek file.')
    if not set(df['class'].unique()).issubset({0,1}):
        raise ValueError('Label bukan 0/1. Periksa sumber file.')
    return df


def sigmoid(z):
    z=np.clip(z,-60,60)
    return 1/(1+np.exp(-z))

@dataclass
class Grad:
    W1:np.ndarray
    b1:np.ndarray
    W2:np.ndarray
    b2:np.ndarray

class MLP:
    """Satu hidden layer sigmoid; output sigmoid; loss binary cross entropy.
       Parameter W1(d,h), b1(1,h), W2(h,1), b2(1,1).
    """
    def __init__(self,input_dim=4,hidden=8,seed=42):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,1/np.sqrt(input_dim),(input_dim,hidden))
        self.b1=np.zeros((1,hidden))
        self.W2=rng.normal(0,1/np.sqrt(hidden),(hidden,1))
        self.b2=np.zeros((1,1))

    def forward(self,X):
        z1=X@self.W1+self.b1
        a1=sigmoid(z1)
        z2=a1@self.W2+self.b2
        p=sigmoid(z2)
        return a1,z2,p

    def predict_proba(self,X):
        return self.forward(X)[2]

    def predict(self,X):
        return (self.predict_proba(X)>=0.5).astype(int)

    def loss(self,X,y):
        _,z2,_=self.forward(X)
        return float(np.mean(np.logaddexp(0,z2)-y*z2))

    def backward(self,X,y):
        N=len(X)
        a1,z2,p=self.forward(X)
        delta2=(p-y)/N
        dW2=a1.T@delta2
        db2=delta2.sum(axis=0,keepdims=True)
        delta1=(delta2@self.W2.T)*a1*(1-a1)
        dW1=X.T@delta1
        db1=delta1.sum(axis=0,keepdims=True)
        return Grad(dW1,db1,dW2,db2)

    def step(self,g,lr):
        self.W1 -= lr*g.W1
        self.b1 -= lr*g.b1
        self.W2 -= lr*g.W2
        self.b2 -= lr*g.b2

    def fit(self,X,y,epochs=300,lr=0.5,eval_data=None,track_every=5):
        history=[]
        for ep in range(1,epochs+1):
            self.step(self.backward(X,y),lr)
            if ep==1 or ep==epochs or ep%track_every==0:
                row={'epoch':ep,'train_loss':self.loss(X,y)}
                if eval_data is not None:
                    Xv,yv=eval_data
                    row['val_loss']=self.loss(Xv,yv)
                    row['val_acc']=float(np.mean(self.predict(Xv)==yv))
                history.append(row)
        return history


def num_grads(model,X,y,eps=1e-5):
    values={}
    for name in ('W1','b1','W2','b2'):
        matrix=getattr(model,name)
        grad=np.zeros_like(matrix)
        for idx in np.ndindex(matrix.shape):
            base=matrix[idx]
            matrix[idx]=base+eps; pos=model.loss(X,y)
            matrix[idx]=base-eps; neg=model.loss(X,y)
            matrix[idx]=base
            grad[idx]=(pos-neg)/(2*eps)
        values[name]=grad
    return values


def gradient_checks():
    model=MLP(input_dim=4,hidden=3,seed=2)
    X=np.array([[0.5,-1.0,0.2,1.3],[-0.2,0.4,1.0,-0.7],[0.3,0.1,-0.5,0.6]])
    y=np.array([[1.],[0.],[1.]])
    analytic=model.backward(X,y)
    approx=num_grads(model,X,y)
    diffs={k:float(np.max(np.abs(getattr(analytic,k)-approx[k]))) for k in approx}
    assert max(diffs.values())<1e-7, f'Gagal gradient check: {diffs}'
    return diffs


def prepare_split(df,seed=42):
    from sklearn.model_selection import train_test_split
    X=df[COLS[:4]].to_numpy(dtype=float)
    y=df['class'].to_numpy(dtype=int).reshape(-1,1)
    # Stratifikasi = proporsi label 0/1 relatif sama pada setiap bagian.
    Xtr,Xrest,ytr,yrest=train_test_split(X,y,test_size=0.40,stratify=y,random_state=seed)
    Xva,Xte,yva,yte=train_test_split(Xrest,yrest,test_size=0.50,stratify=yrest,random_state=seed)
    # Rata-rata dan simpangan baku dihitung HANYA dari data latih.
    mu=Xtr.mean(axis=0)
    sigma=Xtr.std(axis=0)
    sigma=np.where(sigma==0,1,sigma)
    z=lambda x:(x-mu)/sigma
    return (z(Xtr),ytr),(z(Xva),yva),(z(Xte),yte),mu,sigma


def binary_metrics(y_true,y_predict):
    yt=np.asarray(y_true).ravel().astype(int)
    yp=np.asarray(y_predict).ravel().astype(int)
    TP=int(np.sum((yt==1)&(yp==1)))
    TN=int(np.sum((yt==0)&(yp==0)))
    FP=int(np.sum((yt==0)&(yp==1)))
    FN=int(np.sum((yt==1)&(yp==0)))
    return {'accuracy':(TP+TN)/len(yt),'precision_class1':TP/max(1,TP+FP),
            'recall_class1':TP/max(1,TP+FN),'TP':TP,'TN':TN,'FP':FP,'FN':FN}


def save_csv(path,rows):
    if not rows: return
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def evaluate_banknote(df,outdir='hasil_colab',seed=42,benchmark_epochs=120):
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    output=Path(outdir);output.mkdir(exist_ok=True,parents=True)
    (Xtr,ytr),(Xva,yva),(Xte,yte),mu,sigma=prepare_split(df,seed)
    gradient=gradient_checks()
    print('Verifikasi gradien (beda maks):',gradient)
    print(f'Dataset={len(df)} | train={len(Xtr)} val={len(Xva)} test={len(Xte)}')
    print('Jumlah label seluruh data:',df['class'].value_counts().sort_index().to_dict())
    # Sensitivitas: 2 parameter (hidden, learning rate), 3 seed, evaluasi di VAL saja.
    sensit=[]
    for h in (4,8,16):
        for lr in (0.1,0.5,1.0):
            for run_seed in (5,11,23,37,53):
                model=MLP(input_dim=4,hidden=h,seed=run_seed)
                model.fit(Xtr,ytr,epochs=350,lr=lr)
                sensit.append({'hidden':h,'lr':lr,'seed':run_seed,
                              'val_accuracy':float(np.mean(model.predict(Xva)==yva)),
                              'val_loss':model.loss(Xva,yva)})
    save_csv(output/'sensitivitas_raw.csv',sensit)
    grouping=[]
    for h in (4,8,16):
        for lr in (0.1,0.5,1.0):
            rows=[r for r in sensit if r['hidden']==h and r['lr']==lr]
            arr=np.array([r['val_accuracy'] for r in rows])
            grouping.append({'hidden':h,'lr':lr,'val_acc_mean':float(arr.mean()),
                             'val_acc_sd':float(arr.std(ddof=1)),
                             'val_loss_mean':float(np.mean([r['val_loss'] for r in rows]))})
    save_csv(output/'sensitivitas_ringkasan.csv',grouping)
    # Pilih dari VAL; TEST baru dibuka setelah pengaturan dipilih.
    best=sorted(grouping,key=lambda r:(-r['val_acc_mean'],r['val_loss_mean'],r['hidden'],r['lr']))[0]
    Xfinal=np.vstack([Xtr,Xva]); yfinal=np.vstack([ytr,yva])
    model=MLP(input_dim=4,hidden=int(best['hidden']),seed=42)
    history=model.fit(Xfinal,yfinal,epochs=350,lr=float(best['lr']),track_every=1)
    save_csv(output/'kurva_loss.csv',history)
    yp=model.predict(Xte)
    ours=binary_metrics(yte,yp)
    baseline=LogisticRegression(max_iter=500,random_state=42).fit(Xfinal,yfinal.ravel())
    ybase=baseline.predict(Xte)
    other=binary_metrics(yte,ybase)
    print('Konfigurasi pilihan dari VALIDASI:',best)
    print('Hasil TEST (backprop dari nol):',ours)
    print('Hasil TEST (baseline Logistic Regression):',other)
    # 5 ukuran data dari data LATIH, ukuran hidden dan epoch tetap, 5 kali ukur.
    raw=[]
    rng=np.random.default_rng(seed)
    sizes=[max(16,int(len(Xtr)*part)) for part in (0.2,0.4,0.6,0.8,1.)]
    for n in sizes:
        for repeat in range(5):
            take=rng.permutation(len(Xtr))[:n]
            xb,yb=Xtr[take],ytr[take]
            measured=MLP(4,hidden=8,seed=repeat+100)
            # Pemanasan waktu komputasi, tidak mengubah bobot terukur.
            measured.backward(xb,yb)
            start=time.perf_counter()
            measured.fit(xb,yb,epochs=benchmark_epochs,lr=0.5,track_every=benchmark_epochs)
            elapsed=time.perf_counter()-start
            raw.append({'N':n,'ulangan':repeat+1,'epochs':benchmark_epochs,'hidden':8,
                        'waktu_detik':elapsed,'waktu_ms_per_epoch':elapsed*1000/benchmark_epochs})
    save_csv(output/'benchmark_raw.csv',raw)
    summarized=[]
    for n in sizes:
        vals=np.array([r['waktu_ms_per_epoch'] for r in raw if r['N']==n])
        summarized.append({'N':n,'rata2_ms_per_epoch':float(vals.mean()),
                           'sd_ms_per_epoch':float(vals.std(ddof=1)),'pengulangan':len(vals)})
    save_csv(output/'benchmark_ringkasan.csv',summarized)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history])
    ax.set(xlabel='Epoch',ylabel='Binary Cross Entropy',title='Kurva loss data latih + validasi')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'loss_banknote.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    N=np.array([r['N'] for r in summarized]); T=np.array([r['rata2_ms_per_epoch'] for r in summarized])
    sd=np.array([r['sd_ms_per_epoch'] for r in summarized]); ax.errorbar(N,T,yerr=sd,marker='o',capsize=3,label='Pengukuran')
    theoretical=(T[0]/N[0])*N
    ax.plot(N,theoretical,linestyle='--',label='Kurva O(N) diskalakan di titik pertama')
    ax.set(xlabel='Jumlah sampel per langkah pelatihan (N)',ylabel='Waktu per epoch (ms)',title='Benchmark Backpropagation: 5 ukuran x 5 ulangan')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'grafik_benchmark.png',dpi=160);plt.close(fig)
    import sklearn
    report={'status':'EKSPERIMEN_DIJALANKAN_PADA_PERANGKAT_INI',
            'source':'UCI Banknote Authentication (DOI 10.24432/C55P57)',
            'data_rows':len(df),'train_rows':len(Xtr),'valid_rows':len(Xva),'test_rows':len(Xte),
            'feature_names':COLS[:4], 'target_name':'class',
            'class_semantics':'UCI tidak menjelaskan mapping label 0/1 pada halaman dataset; laporkan kelas 0 / kelas 1.',
            'loss':'Binary Cross Entropy','architecture':f'4-{int(best["hidden"])}-1',
            'optimizer':'full-batch Gradient Descent', 'epochs':350,
            'best_hidden':int(best['hidden']),'best_learning_rate':float(best['lr']),
            'gradient_check_max_difference':max(gradient.values()),
            'test_model':ours,'test_baseline_logistic_regression':other,
            'benchmark':summarized,
            'platform':platform.platform(),'python':platform.python_version(),
            'numpy':np.__version__,'sklearn':sklearn.__version__}
    with (output/'ringkasan_hasil.json').open('w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    print('HASIL DISIMPAN KE:',output.resolve())
    return report


def main():
    p=argparse.ArgumentParser(description='Uji Backpropagation pada data Banknote UCI')
    p.add_argument('--data',default=None)
    p.add_argument('--outdir',default='hasil_colab')
    p.add_argument('--benchmark-epochs',type=int,default=120)
    args=p.parse_args()
    df=load_banknote(args.data)
    evaluate_banknote(df,args.outdir,benchmark_epochs=args.benchmark_epochs)

if __name__ == '__main__':
    main()
