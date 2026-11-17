"""Ordinary independent local-region nearest score, with region reuse."""
import numpy as np
from dictionary import encode,decode

def build_catalog(images,model):
    return {key:encode(value,model['codebooks']) for key,value in images.items()}

def search(query,catalog,model,top_k=3):
    q=np.asarray(query,float);out=[]
    for key,codes in catalog.items():
        z=decode(codes,model['codebooks'])
        if len(z)==0:continue
        distances=np.sum((q[:,None,:]-z[None,:,:])**2,axis=2)
        out.append({'id':key,'cost':float(np.mean(np.min(distances,axis=1)))})
    return sorted(out,key=lambda v:(v['cost'],v['id']))[:top_k]
