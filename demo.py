from pathlib import Path
import json
from dictionary import fit
from retrieval import build_catalog,search
if __name__=='__main__':
 d=json.loads((Path(__file__).parent/'fixtures/visual.json').read_text());model=fit(d['training_images'],d['initial'],d['iterations'],d['ridge']);catalog=build_catalog(d['catalog'],model);print(json.dumps({'model':model,'catalog':catalog,'results':search(d['query'],catalog,model)},indent=2))
