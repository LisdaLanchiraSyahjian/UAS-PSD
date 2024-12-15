---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.11.5'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# PREDIKSI HARGA LAPTOP

# Pendahuluan
## Latar Belakang


<p style="text-indent: 50px; text-align: justify;">Kemajuan teknologi yang pesat telah mendorong kebutuhan akan perangkat elektronik, termasuk laptop, sebagai alat pendukung aktivitas sehari-hari, baik dalam pekerjaan, pendidikan, hingga hiburan. Laptop menjadi salah satu perangkat yang paling diminati karena sifatnya yang portabel dan multifungsi. Dengan meningkatnya permintaan akan laptop, produsen menghadirkan berbagai pilihan dengan spesifikasi dan harga yang beragam untuk memenuhi kebutuhan konsumen.</p>
<p style="text-indent: 50px; text-align: justify;">Harga laptop dipengaruhi oleh berbagai faktor, seperti spesifikasi teknis (prosesor, kapasitas penyimpanan, RAM), merek, fitur tambahan, hingga tren pasar dan perkembangan teknologi. Memahami pola harga dan faktor yang memengaruhinya sangat penting, baik bagi konsumen untuk menentukan pilihan yang sesuai dengan anggaran, maupun bagi produsen dan penjual untuk menetapkan strategi pemasaran yang tepat.</p>
<p style="text-indent: 50px; text-align: justify;">Prediksi harga laptop menjadi salah satu aplikasi penting dalam analisis data. Dengan memanfaatkan teknik-teknik seperti machine learning dan analisis statistik, prediksi harga dapat dilakukan berdasarkan data historis dan variabel-variabel relevan. Penelitian ini bertujuan untuk mengidentifikasi faktor-faktor utama yang memengaruhi harga laptop dan membangun model prediksi harga yang akurat. Dengan model ini, diharapkan dapat memberikan wawasan yang berguna bagi konsumen, produsen, dan pelaku pasar untuk membuat keputusan yang lebih informasional.</p>

## Rumusan Masalah
<p style="text-indent: 50px; text-align: justify;">1. Apa saja faktor utama yang memengaruhi harga laptop di pasaran?

2. Bagaimana membangun model prediksi harga laptop yang akurat menggunakan data historis dan variabel terkait?

3. Seberapa efektif model prediksi yang dibangun dalam memberikan wawasan bagi konsumen dan pelaku pasar?</p>

## Tujuan

<p style="text-indent: 50px; text-align: justify;">1. Mengidentifikasi faktor-faktor utama yang memengaruhi harga laptop di pasaran.

2. Membuat model prediksi harga laptop yang akurat menggunakan metode analisis data.</p>

##Data Understanding

<p style="text-indent: 50px; text-align: justify;">pada kali ini saya melakukan prediksi harga alpukat di amerika pada rentang tahun 2015-2018. Dataset yang saya gunakan ini berasal dari kaggle berikut ini linknya:https://www.kaggle.com/datasets/indrawanpratama/laptop pada dataset ini mememiliki beberapa fitur yaitu :

* Nama: Nama atau model laptop yang mengacu pada merek dan seri tertentu. Fitur
ini membantu mengidentifikasi produk secara unik di antara berbagai pilihan di pasaran.

* Tahun: Tahun peluncuran atau produksi laptop. Fitur ini penting untuk memahami relevansi produk dengan teknologi terkini serta tren harga berdasarkan usia produk.

* Harga: Harga jual laptop dalam satuan mata uang tertentu. Fitur ini menjadi target utama dalam prediksi untuk membantu analisis dan pengambilan keputusan.

* Jenis: Jenis atau kategori laptop, misalnya manual, semi automatic, dan automatic. Fitur ini memberikan informasi tentang segmen pasar dan kegunaan utama dari produk tersebut.

* RAM: Kapasitas memori utama (Random Access Memory) yang digunakan oleh laptop, biasanya diukur dalam gigabyte (GB). Fitur ini sangat memengaruhi performa perangkat, terutama dalam menjalankan aplikasi multitugas.

* ROM: Kapasitas penyimpanan internal (Read-Only Memory) atau storage laptop, yang dapat berupa SSD atau HDD, diukur dalam gigabyte (GB) atau terabyte (TB). Fitur ini memengaruhi kapasitas penyimpanan data pengguna.</p>

## Import Library
```{code-cell} python
# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} python
df = pd.read_csv('LAPTOP.csv')
df.head()
```
### Menentukan Missing Value
```{code-cell} python
df = pd.read_csv('LAPTOP.csv')
df.isnull().sum()
```
<p>dapat dilihat tidak terdapat missingvalue</p>

```{code-cell} python
df.info()
print("Shape of data:")
print(df.shape)
```
### Eksplorasi Data Analysis
```{code-cell} python
from google.colab import files
uploaded = files.upload()

# Membaca file setelah diunggah
df = pd.read_excel('LAPTOP.xlsx')
print(df.describe())
```

### Menentukan Otlier
```{code-cell} python
# Mengimpor pustaka
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Membaca dataset (Excel atau CSV sesuai file)
file_path = "LAPTOP.xlsx"  # Ganti sesuai format file Anda
X = pd.read_excel(file_path)

# Memilih hanya kolom numerik
X = X.select_dtypes(include=[float, int])

# Menghapus nilai kosong
X = X.dropna()

# Membuat objek LOF
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)

# Menentukan outlier
y_pred = lof.fit_predict(X)

# Mencetak hasil
print("Predicted table:", y_pred)
print("Negative LOF scores:", -lof.negative_outlier_factor_)

# Menentukan indeks outlier
outlier_indices = [index for index, value in enumerate(y_pred) if value == -1]
print("Data outlier terdapat pada indeks:", outlier_indices)
```
<p>KESIMPULAN: Disini asumsinya adalah 0.1(10%) dari seluruh data dianggap outlier. output -1 adalah ciri-ciri dari outlier. dan output yang menunjukkan angka-angka desimal yang lebih tinggi dari angka-angka desimal lainnya merupakan outliernya. hal ini ditunjukkan pada indeks ke [2, 24, 30, 36, 45]</p>

### Prepocessing 
<p>Preprocessing adalah tahapan dalam proses analisis data yang melibatkan persiapan dan pembersihan data mentah agar siap untuk dianalisis atau dimasukkan ke dalam model. Langkah-langkah ini penting untuk memastikan bahwa data berada dalam format yang sesuai dan bebas dari kesalahan yang bisa mempengaruhi hasil analisis.</p>

### Menghapus Otlier

```{code-cell} python
# Mengimpor pustaka
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Membaca dataset dari file Excel
X = pd.read_excel("LAPTOP.xlsx")

# Memilih hanya kolom numerik dan menghapus nilai kosong
X = X.select_dtypes(include=[float, int]).dropna()

# Membuat objek LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)

# Menentukan status outlier
y_pred = lof.fit_predict(X)

# Menemukan indeks outlier
outlier_indices = [index for index, value in enumerate(y_pred) if value == -1]
print("Data outlier terdapat pada indeks:", outlier_indices)

# Menghapus outlier dari dataset
X_clean = X.drop(index=outlier_indices)

# Menyimpan dataset tanpa outlier ke file baru
X_clean.to_csv("AFF_dataset_tanpa_outlier.csv", index=False)

# Menampilkan jumlah baris asli dan setelah pembersihan
print("Jumlah baris asli:", len(X))
print("Jumlah baris setelah outlier dihapus:", len(X_clean))
print("Dataset tanpa outlier telah disimpan ke 'LAPTOP_tanpa_outlier.csv'")
```
### Pemodelan
```{code-cell} python
import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca data dari file CSV (pastikan 'data.csv' adalah nama file yang benar)
df = pd.read_csv('LAPTOP.csv')

# Memilih kolom fitur (X) dan target (y)
X = df[['nama', 'tahun', 'jenis', 'ram', 'rom']]  # Ganti dengan nama kolom yang sesuai
y = df['harga']  # Ganti dengan nama kolom target yang sesuai

# Membagi data menjadi data pelatihan (80%) dan data pengujian (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan jumlah data pada data pelatihan dan data pengujian
print("Jumlah data pelatihan:", len(X_train))
print("Jumlah data pengujian:", len(X_test))
```

```{code-cell} python
X = df[['nama', 'tahun', 'jenis', 'ram', 'rom']]
y = df['harga']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVM

```{code-cell} python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Normalize the target values to reduce the scale of errors
y = y / np.max(np.abs(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVR model with default parameters
svr = SVR(kernel='rbf')

# Fit the model on the training data
svr.fit(X_train, y_train)

# Predict on the test set
y_pred = svr.predict(X_test)

# Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluasi Model SVR:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")
```

### Random Forest
```{code-cell} python
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Mengonversi data kategorikal dengan OneHotEncoder
# Menyusun pipeline untuk menangani preprocessing dan model dalam satu langkah

# Menentukan kolom yang kategorikal
categorical_columns = ['nama', 'jenis']

# Menggunakan OneHotEncoder untuk mengonversi kolom kategorikal menjadi numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),  # Menambahkan handle_unknown
        ('num', StandardScaler(), ['tahun', 'ram', 'rom'])  # Menstandarisasi kolom numerik
    ])

# Membuat pipeline yang terdiri dari preprocessing dan model SVR
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf'))
])

# Melatih model menggunakan pipeline
model_pipeline.fit(X_train, y_train)

# Prediksi
y_pred_svr = model_pipeline.predict(X_test)

# Evaluasi
print("\nEvaluasi Model SVR:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_svr):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_svr):.4f}")
```
