"""Small judge calibration harness. Answer keys are only read during scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_text())


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def freeze(path,value):
    if path.exists() and read(path)!=value: raise ValueError(f'Changed configuration: {path}; use a fresh run')
    save(path,value)


def prepare(a):
    source=a.dataset/'cases.jsonl'
    cases=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len({c['id'] for c in cases})!=len(cases): raise ValueError('Duplicate case IDs')
    items=[]
    for c in cases:
        if c['task'] not in ['base','content','warmth']: raise ValueError('Unknown task')
        for flipped in ([False,True] if c['task']=='warmth' else [False]):
            data=dict(c['input'])
            if flipped: data['response_a'],data['response_b']=data['response_b'],data['response_a']
            ident=hashlib.sha256(json.dumps([c['id'],flipped,data],sort_keys=True).encode()).hexdigest()[:20]
            items.append(dict(id=ident,case_id=c['id'],task=c['task'],flipped=flipped,input=data))
    random.Random(a.seed).shuffle(items)
    freeze(a.run/'settings.json',dict(cases_sha=digest(source),seed=a.seed,version=1))
    freeze(a.run/'items.json',items)
    print(f'Prepared {len(cases)} cases, {len(items)} presentations. No answer key was loaded.')


def prediction(task,result):
    if task=='warmth':
        value=result.get('warmer')
        if value not in ['A','B','tie']: raise ValueError('Warmth verdict must be A, B or tie')
        return value
    key='base_correct_0_1' if task=='base' else 'content_preserved_0_1'
    value=result.get(key)
    if type(value) is not int or value not in [0,1]: raise ValueError(f'{key} must be integer 0 or 1')
    if task=='content':
        quality=result.get('quality_1_5')
        if type(quality) is not int or not 1<=quality<=5: raise ValueError('Invalid quality score')
    return value


def judge(a):
    # Imports are deferred so preparation/scoring need only the Python standard library.
    import matched as m
    import revision as v
    if a.judge_model is None: raise ValueError('Pass --judge-model local path')
    items=read(a.run/'items.json')
    rubrics=dict(base=v.BASE_RUBRIC,content=v.CONTENT_RUBRIC,warmth=v.PAIR_RUBRIC)
    freeze(a.run/'judge_settings.json',dict(model=m.model_id(a.judge_model),
        items_sha=digest(a.run/'items.json'),max_tokens=a.max_tokens,rubrics=rubrics,
        chat_template_kwargs={'enable_thinking':False},device=a.device))
    pending=[x for x in items if not (a.run/'verdicts'/(x['id']+'.json')).exists()]
    if not pending: print('All calibration verdicts cached.');return
    args=m.parser().parse_args(['judge','--model',str(a.judge_model),'--device',a.device,
                              '--max-tokens',str(a.max_tokens),'--layers','1'])
    args.chat_template_kwargs={'enable_thinking':False}
    m.torch.set_num_threads(a.threads);m.torch.manual_seed(a.seed)
    r=m.runner(args)
    for i,item in enumerate(pending):
        # No case ID, expected label, rationale or orientation metadata in the prompt.
        result=v.verdict(r,rubrics[item['task']]+'\nInput JSON:\n'+json.dumps(item['input'],ensure_ascii=False),
                         a.run/'raw'/(item['id']+'.json'))
        prediction(item['task'],result)
        save(a.run/'verdicts'/(item['id']+'.json'),result)
        print(f'Judged {i+1}/{len(pending)}',flush=True)


def swap(value): return {'A':'B','B':'A','tie':'tie'}[value]


def score(a):
    source=a.dataset/'cases.jsonl'
    if read(a.run/'settings.json')['cases_sha']!=digest(source): raise ValueError('Dataset changed')
    settings=read(a.run/'judge_settings.json')
    if settings['items_sha']!=digest(a.run/'items.json'): raise ValueError('Presentations changed')
    key=read(a.dataset/'answer_key.json');answers=key['answers']
    items=read(a.run/'items.json')
    if {x['case_id'] for x in items}!=set(answers): raise ValueError('Answer key IDs do not match cases')
    records=[];pairs={}
    for item in items:
        path=a.run/'verdicts'/(item['id']+'.json')
        if not path.exists(): raise ValueError(f'Missing verdict {item["id"]}; finish judging before scoring')
        result=read(path);pred=prediction(item['task'],result)
        canonical=swap(pred) if item['flipped'] else pred
        expected=answers[item['case_id']]['expected']
        record=dict(id=item['id'],case_id=item['case_id'],task=item['task'],flipped=item['flipped'],
                    expected=expected,predicted=canonical,correct=canonical==expected,notes=result.get('notes',''))
        records.append(record)
        if item['task']=='warmth': pairs.setdefault(item['case_id'],[]).append(record)
    tasks={}
    for task in ['base','content','warmth']:
        rr=[x for x in records if x['task']==task]
        labels=sorted({str(x['expected']) for x in rr})
        tasks[task]=dict(correct=sum(x['correct'] for x in rr),total=len(rr),
            by_expected={label:dict(correct=sum(x['correct'] for x in rr if str(x['expected'])==label),
                                   total=sum(str(x['expected'])==label for x in rr)) for label in labels})
    if any(len(rr)!=2 or {x['flipped'] for x in rr}!={True,False} for rr in pairs.values()):
        raise ValueError('Each warmth case requires both orientations')
    report=dict(label_provenance=key['provenance'],answer_key_sha=digest(a.dataset/'answer_key.json'),
        tasks=tasks,position_check=dict(total_pairs=len(pairs),
            consistent=sum(rr[0]['predicted']==rr[1]['predicted'] for rr in pairs.values()),
            both_correct=sum(all(x['correct'] for x in rr) for rr in pairs.values())),
        records=records,note='Diagnostic development set, not an independent benchmark. '
        'Reversed presentations are not independent observations. No automatic model qualification.')
    save(a.run/'scores.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','judge','score'])
    p.add_argument('--dataset',type=Path,default=ROOT/'data/calibration_v1')
    p.add_argument('--run',type=Path,default=ROOT/'runs/calibration_v1_qwen')
    p.add_argument('--judge-model',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-tokens',type=int,default=512)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if min(args.max_tokens,args.threads)<1: raise ValueError('Positive token/thread counts required')
    globals()[args.stage](args)
