"""Run E2 and validation, freeze validation-based selection, then evaluate both on test."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print('RUN', *args, flush=True)
    subprocess.run([sys.executable, '-m', *args], cwd=ROOT, check=True)


def main():
    e1 = ROOT/'runs/detect/runs/ppe/e1-opora'
    if not (e1/'provenance.json').exists():
        raise RuntimeError('E1 must complete successfully first')
    run('ppe.cli', 'train', '--config', 'configs/e2.yaml')
    experiments = {'e1': e1, 'e2': ROOT/'runs/detect/runs/ppe/e2-freeze10'}
    results = {}
    for key, folder in experiments.items():
        run('ppe.cli', 'evaluate', '--weights', str(folder/'weights/best.pt'), '--split', 'val', '--name', key+'-val')
        # Ultralytics may prepend its task directory to a relative project path.
        files = list((ROOT/'runs').rglob(key+'-val/metrics.json'))
        if len(files) != 1:
            raise RuntimeError(f'Expected one validation result for {key}: {files}')
        results[key] = json.loads(files[0].read_text())
    selected = max(results, key=lambda k: results[k]['map50_95'])
    selection = {'criterion': 'validation mAP50-95; tie -> E1', 'selected': selected,
                 'validation': {k: v['map50_95'] for k,v in results.items()},
                 'weights': str((experiments[selected]/'weights/best.pt').relative_to(ROOT)),
                 'test_used_for_selection': False}
    (ROOT/'reports/selection.json').write_text(json.dumps(selection,indent=2))
    for key, folder in experiments.items():
        run('ppe.cli', 'evaluate', '--weights', str(folder/'weights/best.pt'), '--split', 'test', '--name', key+'-test')
    run('ppe.analyze', '--weights', selection['weights'], '--split', 'test', '--name', 'selected')


if __name__ == '__main__':
    main()
