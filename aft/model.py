import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def resolve_layers(indices, count=4):
    resolved=tuple(i if i>=0 else count+i for i in indices)
    if not resolved or len(set(resolved))!=len(resolved) or any(i<0 or i>=count for i in resolved):
        raise ValueError('Layers must be unique indices into the four ResNet stages')
    return resolved


class SmallCNN(nn.Module):
    def __init__(self,classes=100,regularization=False):
        super().__init__()
        modules=[]
        for a,b in [(3,64),(64,128),(128,256)]:
            modules.append(nn.Conv2d(a,b,3,padding=1))
            if regularization:modules.append(nn.BatchNorm2d(b))
            modules.extend([nn.ReLU(),nn.MaxPool2d(2)])
        self.features=nn.Sequential(*modules,nn.AdaptiveAvgPool2d(1))
        self.dropout=nn.Dropout(0.3) if regularization else nn.Identity()
        self.classifier=nn.Linear(256,classes)

    def forward_features(self,x):return self.features(x).flatten(1)
    def forward(self,x):return self.classifier(self.dropout(self.forward_features(x)))


def gram(features,kernel='linear',sigma=1.0):
    if sigma<=0:raise ValueError('sigma must be positive')
    features=F.normalize(features,dim=1,eps=1e-6)
    if kernel=='linear':return features@features.T
    if kernel=='rbf':
        distances=torch.cdist(features,features).square()
        return torch.exp(-distances/(2*sigma*sigma))
    raise ValueError('Unknown kernel')


class TransferModel(nn.Module):
    def __init__(self,kernel='linear',sigma=1.0,layers=(-1,),pretrained=True,regularization=False,teacher_size=224):
        super().__init__()
        if kernel not in {'linear','rbf'} or sigma<=0:raise ValueError('Invalid kernel configuration')
        self.layers=resolve_layers(layers);self.kernel=kernel;self.sigma=sigma;self.teacher_size=teacher_size
        self.teacher=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.teacher.requires_grad_(False);self.teacher.eval()
        self.student=SmallCNN(regularization=regularization)
        dims=(64,128,256,512)
        self.selectors=nn.ParameterList([nn.Parameter(torch.zeros(dims[i])) for i in self.layers])
        self.register_buffer('teacher_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('teacher_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.register_buffer('student_mean',torch.tensor([.5071,.4867,.4408]).view(1,3,1,1))
        self.register_buffer('student_std',torch.tensor([.2675,.2565,.2761]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()  # no_grad alone does not freeze BatchNorm running statistics.
        return self

    def teacher_features(self,x):
        x=F.interpolate(x,size=(self.teacher_size,self.teacher_size),mode='bilinear',align_corners=False)
        x=(x-self.teacher_mean)/self.teacher_std
        m=self.teacher;x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
        outputs=[]
        for layer in (m.layer1,m.layer2,m.layer3,m.layer4):
            x=layer(x);outputs.append(F.adaptive_avg_pool2d(x,1).flatten(1))
        return [outputs[i] for i in self.layers]

    def forward(self,x,transfer=True,cached_features=None):
        student=self.student.forward_features((x-self.student_mean)/self.student_std)
        logits=self.student.classifier(self.student.dropout(student))
        if not transfer:return logits,logits.new_zeros(())
        if cached_features is None:
            with torch.no_grad():teacher=self.teacher_features(x)
        else:
            if len(cached_features)!=len(self.layers) or any(t.shape[0]!=x.shape[0] or t.shape[1]!=s.numel() for t,s in zip(cached_features,self.selectors)):
                raise ValueError('Cached teacher feature shape mismatch')
            teacher=[t.detach() for t in cached_features]
        target=gram(student,self.kernel,self.sigma)
        losses=[]
        for features,selector in zip(teacher,self.selectors):
            weighted=features*torch.sigmoid(selector).unsqueeze(0)
            losses.append((target-gram(weighted,self.kernel,self.sigma)).square().mean())
        return logits,torch.stack(losses).mean()
