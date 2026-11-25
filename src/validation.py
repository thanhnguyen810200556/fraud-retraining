"""Final dataset gate. Run from repository root: python -m src.validation."""
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aligned(left, right, key, columns):
    for frame in (left, right):
        require(frame[key].notna().all() and frame[key].is_unique, 'Invalid IDs')
    a = left.set_index(key)[columns].sort_index()
    b = right.set_index(key)[columns].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def run(root=ROOT):
    root = Path(root)
    out = root / 'reports/validation'
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'freeze_manifest.json'
    require(not manifest.exists(), 'Existing freeze manifest: preserve it before creating a new version')
    checks, artifacts = [], {}
    def check(name, fn):
        try:
            fn()
            checks.append(dict(check=name, passed=True, details=''))
        except Exception as e:
            checks.append(dict(check=name, passed=False, details=str(e)))
    def track(p):
        p = str(p).replace('\\', '/')
        path = root / p
        artifacts[p] = dict(sha256=sha256(path), size_bytes=path.stat().st_size)
        return path
    def read(p, **kw):
        return pd.read_parquet(track(p), **kw)
    def load(p):
        return json.loads(track(p).read_text(encoding='utf-8-sig'))
    def csv(p):
        return pd.read_csv(track(p))
    def config(p):
        return yaml.safe_load(track(p).read_text(encoding='utf-8'))
    try:
        cfg = config('configs/dataset.yaml')
        config('configs/features.yaml')
        schema = load('reports/preprocessing/feature_schema.json')
        meta = load('reports/label_delay/label_delay_metadata.json')
        temporal = load('reports/temporal/temporal_metadata.json')
        summary = csv('reports/preprocessing/preprocessing_summary.csv').set_index('dataset')
        def provenance():
            for row in csv('data/raw/dataset_manifest.csv').itertuples():
                require(sha256(track(row.local_path)) == row.sha256, f'Raw hash mismatch: {row.local_path}')
        check('raw provenance hashes', provenance)
        def previous():
            for item in meta['input_artifacts'] + meta['schedule_artifacts'] + [meta['config'], meta['source_code']]:
                require(sha256(track(item['path'])) == item['sha256'], f"Stale artifact: {item['path']}")
            for p, col, expected in [('reports/temporal/temporal_validation.csv','status','passed'), ('reports/label_delay/label_delay_validation.csv','passed',True)]:
                df = csv(p)
                require(len(df)>0 and df[col].eq(expected).all(), p)
        check('previous stage hashes and reports', previous)
        def dataset(name, path, spec, key, time, target, index_path, schedule_paths, ieee):
            df = read(path)
            row = summary.loc[name]
            require(len(df)==row.rows and len(df.columns)==row.columns, 'Shape mismatch')
            require(df[target].isin([0,1]).all() and df[target].sum()==row.fraud_count, 'Target mismatch')
            require(abs(df[target].mean()*100-row.fraud_rate_pct)<1e-8, 'Fraud rate mismatch')
            features = spec['features']
            require(len(features)==row.model_features and set(features)==set(df.columns)-{key,time,target}, 'Feature schema mismatch')
            idx = read(index_path)
            phase_report = csv('reports/temporal/ieee_phase_summary.csv' if ieee else 'reports/temporal/baf_phase_summary.csv')
            if not ieee:
                phase_report = phase_report.loc[phase_report.variant.eq(name.removeprefix('BAF '))]
            merged = df[[key,target]].merge(idx[[key,'phase']],on=key,validate='one_to_one')
            for r in phase_report.itertuples():
                subset = merged.loc[merged.phase.eq(r.phase),target]
                require(len(subset)==r.num_transactions and subset.sum()==r.num_fraud, 'Phase statistics mismatch')
            aligned(df, idx, key, [time])
            require(target not in idx.columns, 'Target in temporal index')
            if ieee:
                day = (idx[time]-cfg['temporal_split']['expected_time_origin'])//86400+1
                require(day.eq(idx.day).all(), 'Day formula mismatch')
                require(day.max()==cfg['temporal_split']['expected_last_day']==temporal['ieee_cis']['last_day'], 'Last day metadata mismatch')
                phase = day.map(lambda d: 'warmup' if d<=30 else 'validation' if d<=86 else 'test' if d<=177 else 'tail')
                block = day.map(lambda d: 0 if d<=30 else (d-31)//7+1 if d<=177 else -1)
            else:
                require(set(idx[time])==set(range(8)), 'Month coverage mismatch')
                phase = idx[time].map(lambda m: 'warmup' if m<=1 else 'validation' if m<=3 else 'test')
                block = idx[time].map(lambda m: 0 if m<=1 else m-1)
            require(phase.eq(idx.phase).all() and block.eq(idx.block_id).all(), 'Phase/block mismatch')
            for p, delay in schedule_paths:
                sch = read(p)
                aligned(idx, sch, key, [time,'phase','block_id'])
                require(not {'isFraud','fraud_bool'}.intersection(sch.columns), 'Target in schedule')
                from src.label_delay import validate_ieee_label_schedule, validate_baf_label_schedule
                validator = validate_ieee_label_schedule if ieee else validate_baf_label_schedule
                require(validator(sch, idx, delay), 'Schedule invalid')
        ip='data/temporal/ieee_temporal_index.parquet'
        for name, suffix in [('FullFeatures','full'),('NoCardID','nocardid')]:
            check('IEEE '+name, lambda name=name,suffix=suffix: dataset('IEEE-CIS '+name, f'data/processed/ieee_{suffix}.parquet',schema['ieee_cis'][name],'TransactionID','TransactionDT','isFraud',ip,[(f'data/simulations/label_schedules/ieee/d{d}.parquet',d) for d in cfg['simulation']['label_delay_days']],True))
        check('IEEE feature set alignment',lambda: aligned(read('data/processed/ieee_full.parquet',columns=['TransactionID','TransactionDT','isFraud']),read('data/processed/ieee_nocardid.parquet',columns=['TransactionID','TransactionDT','isFraud']),'TransactionID',['TransactionDT','isFraud']))
        for name in cfg['datasets']['baf']['variants']:
            slug=Path(name).stem.lower().replace(' ','_')
            check('BAF '+name,lambda name=name,slug=slug: dataset('BAF '+name,f'data/processed/baf/{slug}.parquet',schema['baf'][name],'record_id','month','fraud_bool',f'data/temporal/baf/{slug}_temporal_index.parquet',[(f'data/simulations/label_schedules/baf/{slug}/d{d}.parquet',d) for d in cfg['baf_simulation']['label_delay_months']],False))
        for p in ['src/preprocessing.py','src/temporal_split.py','src/label_delay.py','src/validation.py','requirements.txt']:
            check('track '+p,lambda p=p: track(p))
    except Exception as e:
        checks.append(dict(check='setup',passed=False,details=str(e)))
    pd.DataFrame(checks).to_csv(out/'final_validation.csv',index=False)
    require(all(c['passed'] for c in checks), f'Validation failed; see {out}/final_validation.csv')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src','configs','requirements.txt'],cwd=root,text=True).strip(),'Commit source/config changes first')
    payload=dict(version='1.0.0',status='frozen',created_at=datetime.now(timezone.utc).isoformat(),git_commit=commit,artifacts=artifacts,checks=checks)
    payload['provenance_note'] = 'Source license and upstream version require manual verification; freeze identifies local bytes only.'
    manifest.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(f'Frozen: {manifest}')

if __name__=='__main__':
    run()
