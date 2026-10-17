import torch
from aft.model import TransferModel

def test_cached_features_match_live_loss_and_gradient():
    torch.manual_seed(7);torch.set_num_threads(2)
    m=TransferModel(pretrained=False,kernel='rbf',teacher_size=32).train()
    x=torch.rand(3,3,32,32)
    with torch.no_grad():features=m.teacher_features(x)
    logits,loss=m(x);loss.backward();grad=m.selectors[0].grad.clone();m.zero_grad()
    logits_cached,loss_cached=m(x,cached_features=features);loss_cached.backward()
    assert torch.allclose(logits,logits_cached)
    assert torch.allclose(loss,loss_cached)
    assert torch.allclose(grad,m.selectors[0].grad)
