# Verifikasi file eksperimen aktual (tidak menjalankan ulang pelatihan).
from pathlib import Path
import json, csv
root=Path(__file__).resolve().parents[1]
p=root/'hasil_eksperimen'
with (p/'ringkasan_hasil.json').open() as f: report=json.load(f)
def read(name):
    with (p/name).open(newline='') as f:return list(csv.DictReader(f))
assert report['data_rows']==1372
assert (report['train_rows'],report['valid_rows'],report['test_rows'])==(823,274,275)
assert len(read('sensitivitas_raw.csv'))==45
assert len(read('benchmark_raw.csv'))==25
assert len(read('benchmark_ringkasan.csv'))==5
assert len(read('sensitivitas_ringkasan.csv'))==9
assert abs(report['test_model']['accuracy']-272/275)<1e-12
assert abs(report['test_baseline_logistic_regression']['accuracy']-271/275)<1e-12
assert report['test_model']['TP']+report['test_model']['TN']==272
assert report['test_model']['FP']+report['test_model']['FN']==3
print('Berkas hasil Colab diperiksa: seluruh hitungan dasar konsisten.')
print('Sampel:',report['data_rows'],'| 45 run sensitivitas | 25 run waktu | Test accuracy: {:.2f}%'.format(report['test_model']['accuracy']*100))
