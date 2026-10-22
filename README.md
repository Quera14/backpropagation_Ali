# Analisis Algoritma Backpropagation — Banknote Authentication

Proyek mata kuliah **Algoritma Komputasi**, Program Studi Teknik Informatika, Universitas Hasanuddin. Implementasi Backpropagation jaringan saraf satu hidden layer pada data UCI Banknote Authentication, untuk analisis matematis, validasi gradien, dan pengukuran kompleksitas.

## Isi repositori

| Folder | Isi |
|---|---|
| `src/` | Implementasi inti `banknote_backprop.py` dengan NumPy |
| `notebooks/` | Notebook Google Colab studi kasus asli |
| `hasil_eksperimen/` | Output CSV, JSON, dan PNG dari sesi Colab pemilik proyek |
| `verifikasi_manual/` | Lembar hitung langkah demi langkah + kode pemeriksaan numerik |
| `dokumentasi/` | Protokol eksperimen dan ringkasan bukti |
| `laporan/` | Laporan Word berformat jurnal, disesuaikan dengan contoh PDF |

## Hasil yang dilaporkan

- 1.372 sampel dengan 4 fitur numerik, target dua kelas (0 dan 1).
- Pembagian train/validasi/test: 823 / 274 / 275.
- Model terpilih 4–16–1, learning rate 1.0, 350 epoch.
- Akurasi test Backpropagation **98,91% (272/275)**; baseline Logistic Regression **98,55% (271/275)**.
- Beda maksimum gradient check pada eksperimen: sekitar **9.08 × 10⁻¹²**.
- Benchmark 5 ukuran × 5 ulangan, dengan data mentah tersedia.

## Reproduksi

1. Buka `notebooks/Banknote_Backpropagation_Lengkap_Colab.ipynb` di [Google Colab](https://colab.research.google.com/).
2. Jalankan cell urut. Dataset asli akan diunduh dari UCI, atau unggah file `data_banknote_authentication.txt` dari [UCI](https://archive.ics.uci.edu/dataset/267/banknote+authentication) bila koneksi gagal.
3. Notebook mengekspor hasil ke folder sementara `hasil_colab`. Unduh segera karena Colab tidak menyimpannya permanen saat runtime berakhir.
4. Untuk memeriksa contoh matematika kecil secara terpisah, jalankan `python verifikasi_manual/periksa_gradien_manual.py` setelah menginstal NumPy.
5. Lihat `dokumentasi/DOKUMENTASI_EKSPERIMEN.md` untuk arti angka dan batasan pengukuran.

**Dependensi utama:** NumPy, pandas, matplotlib, scikit-learn. Python 3.13.16 digunakan pada pencatatan asli.

## Keabsahan dan batasan

- File pada folder hasil adalah arsip **pengukuran nyata dari Colab milik mahasiswa**, bukan angka sintetis.
- Contoh hitung kecil merupakan **lembar belajar untuk diverifikasi mahasiswa sendiri**, bukan foto pekerjaan tangan.
- Inti turunan Backpropagation ada pada kode; scikit-learn digunakan untuk pemisahan data dan pembanding Logistic Regression.
- Dataset berisi fitur citra yang telah diekstraksi; tidak memproses foto uang secara langsung.
- Pemetaan label kelas 0/1 ke asli/palsu **tidak diklaim** tanpa dokumen sumber yang memastikannya.
- Tidak ada jaminan bahwa model generalisasi ke uang sungguhan pada kondisi kamera/pemindai lain.

## Sumber dan atribusi

- UCI Machine Learning Repository, Banknote Authentication, DOI: **10.24432/C55P57**. Lisensi/atribusi data mengikuti ketentuan repositori UCI.
- Rumelhart, Hinton, & Williams (1986), *Learning representations by back-propagating errors*, Nature.
- Referensi penelitian tambahan dapat dilihat pada daftar pustaka laporan.
- Penggunaan AI didokumentasikan di lampiran laporan dan harus ditinjau ulang agar akurat menggambarkan bantuan yang benar-benar dipakai.

## Tautan laporan

Laporan DOCX terdapat pada `laporan/Laporan_Backpropagation_Format_Jurnal.docx`. Periksa keterbacaan rumus setelah dibuka dengan Microsoft Word.

## Pemeriksaan konsistensi artefak

Jalankan `python dokumentasi/cek_konsistensi_hasil.py` untuk memeriksa jumlah percobaan dan kecocokan metrik pada JSON dan CSV, tanpa membuat ulang hasil eksperimen. Petunjuk publikasi tersedia pada `PANDUAN_UPLOAD_GITHUB.md`.
