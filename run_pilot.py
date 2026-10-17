"""Matched CIFAR-100 real-data pilot, frozen teacher cache, no augmentation."""
import argparse,copy,hashlib,io,json,platform,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset
from aft.model import TransferModel


def read_split(path):
    data=pq.read_table(path).to_pylist()
    images=np.stack([np.array(Image.open(io.BytesIO(x['img']['bytes'])).convert('RGB')) for x in data])
    return torch.from_numpy(images).permute(0,3,1,2),torch.tensor([x['fine_label'] for x in data])


def stratified(labels,seed):
    generator=torch.Generator().manual_seed(seed);train=[];validation=[]
    for cls in range(100):
        idx=torch.where(labels==cls)[0];idx=idx[torch.randperm(len(idx),generator=generator)]
        train.extend(idx[:100].tolist());validation.extend(idx[100:120].tolist())
    assert not set(train)&set(validation)
    return train,validation


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model,x,y,device):
    model.eval();pred=[];loss=0
    with torch.no_grad():
        for bx,by in DataLoader(TensorDataset(x,y),batch_size=128):
            logits,_=model(bx.to(device),transfer=False)
            loss+=torch.nn.functional.cross_entropy(logits,by.to(device),reduction='sum').item()
            pred.extend(logits.argmax(1).cpu().tolist())
    return {'accuracy':float((torch.tensor(pred)==y).float().mean()),'loss':loss/len(y),'examples':len(y)},pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',default='data/cifar100-mirror');p.add_argument('--device',default='mps');p.add_argument('--epochs',type=int,default=5);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device(a.device)
    train_raw,train_labels=read_split(Path(a.data)/'train.parquet')
    test_raw,test_labels=read_split(Path(a.data)/'test.parquet')
    assert len(train_labels)==50000 and len(test_labels)==10000
    tr_idx,val_idx=stratified(train_labels,42)
    x=train_raw[tr_idx].float()/255;y=train_labels[tr_idx]
    vx=train_raw[val_idx].float()/255;vy=train_labels[val_idx]
    tx=test_raw.float()/255
    del train_raw,test_raw
    teacher=TransferModel(kernel='rbf',sigma=1,layers=(-1,),pretrained=True,teacher_size=224).to(device).eval()
    initial={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    manifest={'kind':'real_cifar100_pilot','seed':a.seed,'split_seed':42,'epochs':a.epochs,'train_per_class':100,'validation_per_class':20,'test_examples':10000,'batch_size':64,'optimizer':'Adam','lr':.001,'kernel':'rbf','sigma':1,'layers':[-1],'augmentation':False,'beta':{'baseline':0,'aft_rbf':1},'device':str(device),'torch':torch.__version__,'python':platform.python_version(),'initial_state_sha256':tensor_hash(initial),'train_indices':tr_idx,'validation_indices':val_idx,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('aft/model.py')]},'dataset_sources':[json.loads(p.read_text()) for p in sorted(Path(a.data).glob('*.source.json'))],'determinism':'warn_only; MPS does not guarantee bitwise reproducibility','selection':'best validation accuracy within each configured run; test evaluated once per selected checkpoint'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cached=[]
    with torch.no_grad():
        for i in range(0,len(x),64):
            cached.append(teacher.teacher_features(x[i:i+64].to(device))[0].cpu())
            if i%1024==0:print('cache',i,'/',len(x),flush=True)
    cache=torch.cat(cached);torch.save(cache,out/'teacher_features.pt')
    manifest['cache_sha256']=hashlib.sha256((out/'teacher_features.pt').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('cache complete, seconds',time.time()-start,flush=True)
    del teacher
    results={}
    for name,beta in [('baseline',0),('aft_rbf',1)]:
        torch.manual_seed(a.seed)
        model=TransferModel(kernel='rbf',sigma=1,layers=(-1,),pretrained=False,teacher_size=224).to(device)
        model.load_state_dict(initial);assert tensor_hash(model.state_dict())==manifest['initial_state_sha256']
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        generator=torch.Generator().manual_seed(a.seed)
        loader=DataLoader(TensorDataset(x,y,cache),batch_size=64,shuffle=True,generator=generator)
        history=[];best=-1;run_start=time.time()
        for epoch in range(a.epochs):
            model.train();loss_sum=align_sum=correct=total=0
            for bx,by,bf in loader:
                bx,by,bf=bx.to(device),by.to(device),bf.to(device)
                optimizer.zero_grad();logits,alignment=model(bx,transfer=bool(beta),cached_features=[bf] if beta else None)
                task=torch.nn.functional.cross_entropy(logits,by);loss=task+beta*alignment
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();optimizer.step()
                total+=len(by);correct+=(logits.argmax(1)==by).sum().item();loss_sum+=loss.item()*len(by);align_sum+=alignment.item()*len(by)
            val,_=evaluate(model,vx,vy,device)
            record={'epoch':epoch+1,'train_accuracy':correct/total,'train_loss':loss_sum/total,'alignment_loss':align_sum/total,'validation':val,'elapsed_seconds':time.time()-run_start}
            history.append(record);print(name,json.dumps(record),flush=True)
            (out/(name+'_history.json')).write_text(json.dumps(history,indent=2))
            if val['accuracy']>best:
                best=val['accuracy'];best_epoch=epoch+1;torch.save(model.state_dict(),out/(name+'_best.pt'))
        model.load_state_dict(torch.load(out/(name+'_best.pt'),weights_only=True,map_location=device))
        test,pred=evaluate(model,tx,test_labels,device)
        results[name]={'history':history,'best_epoch':best_epoch,'test':test,'training_seconds':time.time()-run_start}
        (out/(name+'_test_predictions.json')).write_text(json.dumps({'predictions':pred,'labels':test_labels.tolist()}))
        (out/'results.json').write_text(json.dumps(results,indent=2));print(name,'TEST',json.dumps(test),flush=True)
        del model,optimizer
    results['wall_seconds']=time.time()-start
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('COMPLETE',results['wall_seconds'],flush=True)

if __name__=='__main__':main()
