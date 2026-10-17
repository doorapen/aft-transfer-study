import torch
import pytest
from aft.model import TransferModel,gram,resolve_layers
from train import split_indices,epoch
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(2)


def test_negative_layers_are_nonempty_and_validated():
    assert resolve_layers([-4,-3,-2,-1])==(0,1,2,3)
    for indices in [[],[-5],[0,-4]]:
        with pytest.raises(ValueError):resolve_layers(indices)


def test_frozen_teacher_bn_and_student_gradients():
    torch.manual_seed(0)
    m=TransferModel(pretrained=False,layers=(-4,-3,-2,-1),teacher_size=32)
    before={k:v.clone() for k,v in m.teacher.state_dict().items()}
    m.train();assert not m.teacher.training
    logits,alignment=m(torch.rand(4,3,32,32))
    assert alignment.item()>0
    (logits.square().mean()+alignment).backward()
    assert all(p.grad is None for p in m.teacher.parameters())
    assert m.student.classifier.weight.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.selectors)
    assert all(torch.equal(v,before[k]) for k,v in m.teacher.state_dict().items())


def test_kernel_and_split():
    x=torch.randn(5,7,requires_grad=True);k=gram(x,'rbf')
    assert torch.allclose(k,k.T) and torch.allclose(k.diag(),torch.ones(5))
    with pytest.raises(ValueError):gram(x,'rbf',0)
    a,b=split_indices(100,20,42)
    assert not set(a)&set(b) and len(a)+len(b)==100
    assert (a,b)==split_indices(100,20,42)


def test_evaluation_does_not_update_model():
    m=TransferModel(pretrained=False,teacher_size=32)
    before={k:v.clone() for k,v in m.state_dict().items()}
    loader=DataLoader(TensorDataset(torch.rand(5,3,32,32),torch.zeros(5,dtype=torch.long)),batch_size=3)
    metrics=epoch(m,loader,torch.device('cpu'),0)
    assert metrics['examples']==5 and metrics['alignment_loss']==0
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
