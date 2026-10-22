# Dokumentasi eksperimen aktual Google Colab

Sumber angka adalah **arsip hasil asli yang diberikan pemilik proyek**: `Hasil_Banknote_Colab (1).zip`. File CSV/JSON/PNG di folder `hasil_eksperimen` disalin tanpa mengubah isinya.

## Data dan lingkungan

- Dataset: **UCI Banknote Authentication** (DOI: 10.24432/C55P57); 1.372 baris, fitur variance, skewness, curtosis, entropy, label 0/1.
- Train/validasi/test: **823 / 274 / 275**; pemisahan stratified. Standardisasi dipelajari dari train.
- Mesin: **Linux-6.6.122+-x86_64-with-glibc2.39**, Python **3.13.16**, NumPy **2.1.3**, scikit-learn **1.6.1**. Identitas CPU/GPU tidak tersedia dari log, sehingga tidak diisi berdasarkan tebakan.
- Inti: `src/banknote_backprop.py`, algoritma Backpropagation dan update full-batch dengan NumPy.
- Data label 0/1 tetap disebut *kelas 0/kelas 1* (tidak diasumsikan pemetaan asli/palsunya).

## Protokol eksperimen

- H ∈ {4,8,16} dan learning rate ∈ {0.1,0.5,1.0}, masing-masing 5 seed: **45 percobaan validasi**.
- Model terpilih setelah validasi: **4-16-1**, LR **1.0**, epoch **350**.
- Hasil pengujian akhir: akurasi Backpropagation **98.91% (272/275)**; Logistic Regression **98.55% (271/275)**.
- TP/TN/FP/FN Backpropagation (kelas 1 positif): 122/150/3/0.
- Pemeriksaan gradien pada eksperimen: beda maksimum 9.0781340778e-12.
- Benchmark: 5 ukuran sampel × 5 pengulangan. Waktu ditampilkan dalam milidetik per epoch; pengaturan benchmark berbeda dari model final (H=8, 120 epoch).

## Hasil benchmark berdasarkan berkas asli

| N (sampel) | Mean (ms/epoch) | SD (ms/epoch) | Ulangan |
|---:|---:|---:|---:|
| 164 | 0.151511 | 0.006492 | 5 |
| 329 | 0.134106 | 0.037975 | 5 |
| 493 | 0.151849 | 0.021470 | 5 |
| 658 | 0.169323 | 0.001113 | 5 |
| 823 | 0.199549 | 0.006798 | 5 |

## Berkas pendukung

- `ringkasan_hasil.json`: konfigurasi, metrik, jumlah data, platform.
- `sensitivitas_raw.csv`: setiap percobaan parameter.
- `sensitivitas_ringkasan.csv`: agregat hasil validasi.
- `benchmark_raw.csv`: semua 25 pencatatan waktu.
- `benchmark_ringkasan.csv`: lima agregat waktu.
- `kurva_loss.csv` dan `loss_banknote.png`: kurva loss pelatihan model akhir.
- `grafik_benchmark.png`: perbandingan waktu pengukuran dan kurva linear beracuan titik pertama.

**Catatan ilmiah:** pengukuran CPU kecil dapat didominasi overhead Python/NumPy; garis linear ideal pada plot bukan regresi atas hasil, sehingga tidak boleh diperlakukan sebagai bukti empiris ketat. Rekaman eksekusi asli telah dilakukan oleh pemilik proyek di Google Colab; file pada repo memungkinkan pemeriksaan ulang hasil tersebut.
