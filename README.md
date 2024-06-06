# Backpropagation-Momentum
Berikut adalah deskripsi untuk README pada repository GitHub Anda:

---

# Prediksi Produksi Data Batik (2001-2020)

Proyek ini bertujuan untuk memprediksi produksi empat jenis batik dari tahun 2001 hingga 2020. Jenis batik yang dianalisis adalah:

1. Produksi Batik Cap
2. Produksi Batik Tulis
3. Produksi Batik Semitulis
4. Produksi Batik Semiwarna

## Pendahuluan

Dalam proyek ini, setiap jenis batik dianalisis secara terpisah. Data produksi dipecah menjadi beberapa hari, yaitu h-4, h-3, h-2, h-1, dan h. Tujuan dari proyek ini adalah memprediksi produksi pada hari ke-h berdasarkan data dari hari-hari sebelumnya.

## Data

Data yang digunakan mencakup produksi batik dari tahun 2001 hingga 2020. Setiap dataset batik memiliki struktur yang sama, yaitu terdiri dari data produksi untuk hari ke-h yang diprediksi menggunakan data produksi dari empat hari sebelumnya (h-4, h-3, h-2, h-1).

## Pemodelan

Pemodelan dilakukan secara terpisah untuk setiap jenis batik. Sebelum memulai pemodelan, data untuk setiap jenis batik dipisahkan dan diolah secara individual. Pembagian data dilakukan dengan rasio 70% untuk training dan 30% untuk testing.

## Langkah-Langkah

1. **Persiapan Data:** 
   - Memisahkan data produksi untuk setiap jenis batik.
   - Membuat dataset dengan struktur h-4, h-3, h-2, h-1, h.

2. **Pembagian Data:**
   - Data dibagi menjadi data training (70%) dan data testing (30%).

3. **Pemodelan:**
   - Model dibuat secara terpisah untuk setiap jenis batik.
   - Training model menggunakan data training.
   - Evaluasi model menggunakan data testing.

## Hasil

Hasil dari prediksi untuk masing-masing jenis batik akan dianalisis dan dibandingkan untuk menentukan akurasi model dalam memprediksi produksi pada hari ke-h.

## Penggunaan

1. Clone repository ini:
   ```bash
   git clone https://github.com/username/repo-name.git
   ```
2. Install dependencies yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan script pemodelan:
   ```bash
   python modeling_script.py
   ```

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan fork repository ini, buat branch baru untuk fitur atau perbaikan Anda, dan kirim pull request. Semua kontribusi sangat dihargai!

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

Dengan deskripsi ini, README Anda akan memberikan panduan yang jelas dan informatif kepada pengguna atau pengembang lain yang tertarik dengan proyek Anda.
