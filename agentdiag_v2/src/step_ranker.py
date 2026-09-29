"""Sparse step ranker; prediction uses only observed trace messages."""

import math,re,zlib
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

def hashes(t,base,size):
 words=re.findall('[a-z][a-z0-9_]{2,}',t.lower())[:400]
 c=Counter(zlib.crc32(w.encode())%size for w in words)
 v={base+i:math.log1p(min(n,8)) for i,n in c.items()}
 norm=math.sqrt(sum(x*x for x in v.values()))
 return {i:x/norm for i,x in v.items()} if norm else {}

D=4096+2048+256+32
def features(record,j):
 msgs=record['messages'];m=msgs[j];n=len(msgs)
 f=hashes(m['content'],0,4096)
 if j+1<n:f.update(hashes(msgs[j+1]['content'],4096,2048))
 f.update(hashes(m['agent'].replace('_',' '),6144,256))
 a=m['agent'];agent_idx=record['agents'].index(a)
 prev=msgs[j-1] if j else None;nxt=msgs[j+1] if j+1<n else None
 authored=[q for q in msgs if q['agent']==a]
 ai=authored.index(m)
 numeric=[
  j/max(n-1,1),m['raw_index']/max(record['raw_message_count']-1,1),
  1/(j+1),min(j/20,1),min((n-1-j)/20,1),
  float(j==0),float(j==1),float(j==2),float(j==3),float(j==4),float(j==5),
  float(6<=j<=9),float(j>=10),
  float(ai==0),float(ai==1),float(ai==len(authored)-1),
  ai/max(len(authored)-1,1),agent_idx/max(len(record['agents'])-1,1),
  min(math.log1p(len(m['content']))/10,1.5),
  float(m['content'].strip().lower().startswith('you are given:')),
  float(m['content'].strip().upper()=='TERMINATE'),
  float('terminal' in a.lower() or 'computer' in a.lower()),
  m['flags']['error'],m['flags']['uncertain'],m['flags']['correction'],
  float(prev is not None and prev['flags']['error']),
  float(nxt is not None and nxt['flags']['error']),
  float(nxt is not None and nxt['flags']['correction']),
  float(prev is not None and prev['agent'] != a),
  float(nxt is not None and nxt['agent'] != a),
  float(nxt is not None and nxt['agent']==a),
  float('verification' in a.lower() or 'validator' in a.lower()),
 ]
 f.update({6400+i:v for i,v in enumerate(numeric) if v})
 return f

def design(records):
 data=[];indices=[];indptr=[0];starts=[];lengths=[]
 for r in records:
  starts.append(len(indptr)-1);lengths.append(len(r['messages']))
  for j in range(len(r['messages'])):
   for i,v in sorted(features(r,j).items()):indices.append(i);data.append(v)
   indptr.append(len(data))
 return csr_matrix((np.array(data),np.array(indices),np.array(indptr)),shape=(len(indptr)-1,D)),np.array(starts),np.array(lengths)

def fit(records,l2):
 x,starts,lengths=design(records)
 labels=[next((j for j,m in enumerate(r['messages']) if m['raw_index']==r['mistake_step']
               and m['agent']==r['culprit_name']),None) for r in records]
 if None in labels:raise ValueError('unmatched step')
 correct=starts+np.array(labels)
 def obj(w):
  scores=np.asarray(x@w).ravel();scores-=np.repeat(np.maximum.reduceat(scores,starts),lengths)
  e=np.exp(scores);p=e/np.repeat(np.add.reduceat(e,starts),lengths)
  loss=-np.log(p[correct].clip(1e-12)).sum()+l2*(w@w)/2
  p[correct]-=1;grad=np.asarray(x.T@p).ravel()+l2*w
  return loss,grad
 res=minimize(obj,np.zeros(D),method='L-BFGS-B',jac=True,options={'maxiter':100})
 if not res.success and res.status!=1:raise RuntimeError(res.message)
 return {'weights':res.x.tolist(),'l2':l2,'training_traces':len(records),'dimension':D}

def predict(model,records):
 x,starts,lengths=design(records)
 w=np.asarray(model['weights'])
 s=np.asarray(x@w).ravel();s-=np.repeat(np.maximum.reduceat(s,starts),lengths)
 e=np.exp(s);p=e/np.repeat(np.add.reduceat(e,starts),lengths)
 output=[]
 for record,start,length in zip(records,starts,lengths):
  step_probs=p[start:start+length]
  agents=record['agents']
  agent_probs=[float(sum(step_probs[j] for j,m in enumerate(record['messages']) if m['agent']==a)) for a in agents]
  selected=int(np.argmax(step_probs))
  output.append({'predicted_step':record['messages'][selected]['raw_index'],
                 'predicted_agent':agents[int(np.argmax(agent_probs))],
                 'agent_probabilities':agent_probs,'step_probabilities':step_probs.tolist()})
 return output
