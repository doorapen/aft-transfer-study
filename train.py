"""Train/validate, select on validation only, then evaluate test once."""
import argparse
import hashlib
import json
import random
import platform
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset,TensorDataset
from torchvision import datasets,transforms
from aft.model import TransferModel


def split_indices(n,val_size,seed):
    if not 0<val_size<n:raise ValueError('Validation size must be between zero and training size')
    indices=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[val_size:],indices[:val_size]


def epoch(model,loader,device,beta,optimizer=None):
    model.train(optimizer is not None)
    total=correct=0;loss_sum=task_sum=alignment_sum=0.0
    for inputs,labels in loader:
        inputs,labels=inputs.to(device),labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits,alignment=model(inputs,transfer=beta>0)
            task=nn.functional.cross_entropy(logits,labels)
            loss=task+beta*alignment
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad();loss.backward();optimizer.step()
        n=len(labels);total+=n;correct+=(logits.argmax(1)==labels).sum().item()
        loss_sum+=loss.item()*n;task_sum+=task.item()*n;alignment_sum+=alignment.item()*n
    if not total:raise ValueError('Empty data loader')
    return {'examples':total,'accuracy':correct/total,'loss':loss_sum/total,'task_loss':task_sum/total,'alignment_loss':alignment_sum/total}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='data');p.add_argument('--download',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Synthetic images, random teacher, no benchmark claim')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if cfg['epochs']<1 or cfg['batch_size']<2 or cfg['beta']<0:raise ValueError('Invalid config')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device=torch.device(cfg.get('device','cpu'))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if a.smoke:
        training=TensorDataset(torch.rand(24,3,32,32),torch.randint(0,100,(24,)))
        train_idx,val_idx=split_indices(24,8,cfg['seed'])
        trainset,valset=Subset(training,train_idx),Subset(training,val_idx)
        testset=TensorDataset(torch.rand(8,3,32,32),torch.randint(0,100,(8,)))
    else:
        augment=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()] if cfg.get('augmentation') else []
        training=datasets.CIFAR100(a.data,train=True,download=a.download,transform=transforms.Compose(augment+[transforms.ToTensor()]))
        validation=datasets.CIFAR100(a.data,train=True,download=False,transform=transforms.ToTensor())
        train_idx,val_idx=split_indices(len(training),cfg['validation_size'],cfg['seed'])
        trainset,valset=Subset(training,train_idx),Subset(validation,val_idx)
        testset=datasets.CIFAR100(a.data,train=False,download=False,transform=transforms.ToTensor())
    loaders=[DataLoader(ds,batch_size=cfg['batch_size'],shuffle=i==0,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed'])) for i,ds in enumerate([trainset,valset,testset])]
    model=TransferModel(kernel=cfg['kernel'],sigma=cfg.get('sigma',1),layers=cfg['layers'],pretrained=not a.smoke,regularization=cfg.get('regularization',False),teacher_size=32 if a.smoke else 224).to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])
    manifest={'config':cfg,'mode':'synthetic_smoke' if a.smoke else 'cifar100','python':platform.python_version(),'torch':torch.__version__,'teacher_weights':'random' if a.smoke else 'ResNet18_Weights.DEFAULT','teacher_input_size':32 if a.smoke else 224,'train_indices':train_idx,'validation_indices':val_idx,'test_selection':'official test split, evaluated after validation selection','config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    best=-1;history=[]
    for i in range(1 if a.smoke else cfg['epochs']):
        tr=epoch(model,loaders[0],device,cfg['beta'],optimizer)
        val=epoch(model,loaders[1],device,0)
        history.append({'epoch':i+1,'train':tr,'validation':val})
        if val['accuracy']>best:
            best=val['accuracy'];torch.save(model.state_dict(),out/'best.pt');best_epoch=i+1
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location=device))
    test=epoch(model,loaders[2],device,0)
    result={'mode':manifest['mode'],'best_epoch':best_epoch,'history':history,'test':test}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
