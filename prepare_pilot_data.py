"""Download public CIFAR-100 Parquet mirror; preserve source and content hashes."""
import hashlib,json,urllib.request
from pathlib import Path

root=Path('data/cifar100-mirror');root.mkdir(parents=True,exist_ok=True)
for split in ['train','test']:
    path=root/(split+'.parquet');source=path.with_suffix('.source.json')
    if path.exists():
        if not source.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=json.loads(source.read_text())['sha256']:
            raise ValueError('Existing data lacks matching provenance')
        continue
    url=f'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{split}-00000-of-00001.parquet'
    content=urllib.request.urlopen(url,timeout=120).read()
    path.write_bytes(content)
    source.write_text(json.dumps({'url':url,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()},indent=2))
    print(split,len(content))
