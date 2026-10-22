# Lembar verifikasi perhitungan Backpropagation — contoh 2–2–1

**Tujuan:** memberikan langkah hitungan yang dapat diperiksa secara manual dan dicocokkan dengan skrip numerik. **Ini bukan hasil tulisan tangan mahasiswa**, dan tidak boleh diklaim demikian. Mahasiswa harus memeriksa setiap langkah dan menandai bagian yang sudah dikerjakan sendiri.

**Penting:** contoh ini memakai **Binary Cross Entropy (BCE)**, sama seperti eksperimen Banknote utama. Data pada contoh merupakan angka ilustrasi dan **bukan salah satu sampel Banknote**.

Parameter: `x = [1, 2]`, `y = 1`, `W1 = [[0.1, 0.3], [0.2, 0.4]]`, `b1 = [0.1, 0.1]`, `W2 = [0.5, -0.4]^T`, `b2 = 0`, `η = 1`.

## A. Forward propagation

1. `z1 = x W1 + b1 = [1×0.1 + 2×0.2 + 0.1, 1×0.3 + 2×0.4 + 0.1] = [0.6, 1.2]`.
2. Sigmoid: `a1 = [sigmoid(0.6), sigmoid(1.2)] = [0.645656, 0.768525]`.
3. `z2 = 0.645656×0.5 + 0.768525×(-0.4) = 0.015418`.
4. `p = sigmoid(z2) = 0.503854`.
5. `BCE = log(1+exp(z2)) - y×z2 = 0.685468`.

## B. Backpropagation

1. Delta output: `δ2 = (p-y)/N = -0.496146`, dengan `N=1`.
2. Gradien bobot output: `dW2 = a1ᵀ δ2 = [-0.320339, -0.381300]ᵀ` dan `db2 = -0.496146`.
3. Faktor turunan sigmoid pada hidden: `a1(1-a1) = [0.228784, 0.177894]`.
4. Delta hidden: `δ1 = (δ2 W2ᵀ) ⊙ a1 ⊙ (1-a1) = [-0.056755, 0.035305]`.
5. Gradien bobot hidden: `dW1 = xᵀ δ1 = [[-0.056755, 0.035305], [-0.113510, 0.070609]]`; `db1 = δ1`.

## C. Satu langkah perubahan parameter

Dengan rumus `parameter_baru = parameter_lama − η × gradien`:

- `W1 baru = [[0.156755, 0.264695], [0.313510, 0.329391]]`.
- `b1 baru = [0.156755, 0.064695]`.
- `W2 baru = [0.820339, -0.018700]ᵀ`; `b2 baru = 0.496146`.
- Forward ulang: `p baru = 0.745074`; `loss baru = 0.294271`.

Loss turun pada langkah ilustrasi ini. Penurunan sekali langkah bukan jaminan konvergensi global.

## D. Pemeriksaan gradien numerik

Untuk setiap bobot `θ`, gunakan beda tengah: `(L(θ+ε)-L(θ-ε))/(2ε)` dengan `ε = 1e-5`. Bandingkan dengan `dW1`, `db1`, `dW2`, `db2` di atas. Batas pemeriksaan skrip adalah `1e-8`.

**Bukti kode:** [`periksa_gradien_manual.py`](periksa_gradien_manual.py) dan [`hasil_pemeriksaan_numerik.txt`](hasil_pemeriksaan_numerik.txt). Pemeriksaan gradien pada dataset asli ada dalam `hasil_eksperimen/ringkasan_hasil.json` (beda maksimum `9.078134077800115e-12`).

## Daftar pemeriksaan oleh mahasiswa

- [ ] Saya dapat menghitung `z1`, `a1`, `z2`, `p`, dan BCE tanpa melihat kode.
- [ ] Saya dapat menurunkan `δ2 = p-y` untuk sigmoid + BCE.
- [ ] Saya dapat menunjukkan asal perkalian pada `dW1` dan `dW2`.
- [ ] Saya menjalankan sendiri skrip dan cocokkan hasilnya dengan tabel.
- [ ] Saya bisa menjelaskan perbedaan verifikasi manual, finite differences, dan pengujian Banknote asli.

**Nama pemeriksa / tanggal / tanda tangan (isi setelah benar-benar diperiksa):** _________________________
