"""Working ordinary sequential residual quantization."""
import numpy as np

def encode(vectors,codebooks):
    c=np.asarray(codebooks,float);x=np.asarray(vectors,float);codes=[]
    for row in x:
        residual=row.copy();code=[]
        for book in c:
            j=int(np.argmin(np.sum((book-residual)**2,axis=1)));code.append(j);residual-=book[j]
        codes.append(code)
    return codes

def decode(codes,codebooks):
    c=np.asarray(codebooks,float)
    return np.array([sum((c[l,j] for l,j in enumerate(code)),np.zeros(c.shape[2])) for code in codes])

def fit(images,initial,iterations=2,ridge=0.1):
    c=np.array(initial,float,copy=True);x=np.array([v for im in images for v in im],float)
    for _ in range(iterations):
        residual=x.copy()
        for l in range(len(c)):
            ids=np.argmin(np.sum((residual[:,None,:]-c[l][None,:,:])**2,axis=2),axis=1)
            for k in range(len(c[l])):
                if np.any(ids==k):c[l,k]=residual[ids==k].mean(0)
            residual-=c[l,ids]
    return {'codebooks':c.tolist()}
