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

# PREDIKSI HARGA TEPUNG TERIGU

<P> Saya mengambil data prediksi harga tepung terigu

https://www.kaggle.com/datasets/nirwana22/harga-bahan-sembako  yang selanjutnya akan saya uji sebagai Proyek Sains Data. Untuk lebih jelasnya dapat disimak langkah-langkah dalam mengerjakan proyek ini :</P>

## Data Undesstanding
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px

color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

```

```{code-cell} python

import pandas as pd

# Ubah URL ke URL raw GitHub
url = "https://raw.githubusercontent.com/LisdaLanchiraSyahjian/Proyek-Sain-Data/main/Tepung-Terigu.xlsx"

# Membaca file Excel dari URL raw GitHub
df = pd.read_excel(url)

# Mengurutkan ulang data jika diperlukan (misalnya dari yang terbaru ke terlama)
df = df.iloc[::-1].reset_index(drop=True)

# Menampilkan 10 baris pertama dari data
df.head(32)
```

```{code-cell} python

import pandas as pd

# Menampilkan daftar nama kolom untuk memastikan nama kolom yang benar
print(df.columns)

# Misalkan kolom 'Week' adalah 'Tanggal' dalam dataset, perbaiki nama kolom
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

# Menampilkan 5 baris pertama untuk verifikasi
df.head()
```
### Ploting Data
```{code-cell} python
import matplotlib.pyplot as plt
import pandas as pd

# Pastikan kolom 'Tanggal' adalah datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

# Pastikan tidak ada nilai NaN di kolom yang diperlukan untuk plot
df = df.dropna(subset=['Tanggal', 'Harga'])

# Membuat visualisasi harga tepung terigu dari waktu ke waktu
plt.figure(figsize=(20,6))

# Plot data tanggal (Tanggal) dan harga Tepung Terigu (Harga)
plt.plot(df["Tanggal"], df['Harga'], lw=2)

# Pengaturan label sumbu dan judul
plt.xlabel("Tanggal", fontsize=16)
plt.ylabel("Harga Tepung Terigu (Rp)", fontsize=16)
plt.title("Perkembangan Harga Tepung Terigu dari Waktu ke Waktu", fontsize=16)

# Menampilkan plot
plt.show()
```
```{code-cell} python
df.describe()
```
### Ekstraksi Fitur
```{code-cell} python
import pandas as pd

# Membuat pergeseran data pada kolom 'Harga'
df_slide = df.copy()  # Menghindari modifikasi langsung pada df asli
df_slide['xt-3'] = df_slide['Harga'].shift(-3)
df_slide['xt-2'] = df_slide['Harga'].shift(-2)
df_slide['xt-1'] = df_slide['Harga'].shift(-1)
df_slide['xt'] = df_slide['Harga']

# Menghapus kolom asli 'Harga'
df_slide = df_slide.drop(columns=['Harga'])

# Menampilkan 5 baris pertama dari dataframe yang sudah diubah
df_slide.head()
```
```{code-cell} python
df_slide.dtypes
```
## Prepocessing
### Menentukan Missing Value
```{code-cell} python
df_slide_cleaned = df_slide.dropna()
df_slide_cleaned.isna().sum()
```
<p>Tidak Terdapat missing value</p>

### Normalisasi
```{code-cell} python
# Cek apakah kolom selain 'Week' memang berisi NaN sebelum normalisasi
print("Data sebelum normalisasi:")
print(df_slide_cleaned.head())
```
```{code-cell} python
from sklearn.preprocessing import MinMaxScaler

data = {
    'Tanggal': ['2020-01-31','2020-01-30','2020-01-29','2020-01-28','2020-01-27'],
    'xt-3': [9624.0,9612.0,9000.0,10237.0,9620.0],
    'xt-2': [9568.0,9624.0,9612.0,9000.0,10237],
    'xt-1': [9629,9568,9624,9612,9000],
    'xt': [9603,9629,9568,9624,9612]
}

df_slide_cleaned = pd.DataFrame(data)
df_slide_cleaned['Tanggal'] = pd.to_datetime(df_slide_cleaned['Tanggal'])  # Pastikan kolom Tanggal berbentuk datetime jika dibutuhkan

# Memisahkan kolom 'Tanggal' dari data yang akan dinormalisasi
week_column = df_slide_cleaned['Tanggal']
data_to_normalize = df_slide_cleaned.drop(columns=['Tanggal'])

# Normalisasi Min-Max pada data selain 'Tanggal'
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# Menambahkan kembali kolom 'Tanggal' yang asli
df_normalized = pd.concat([week_column.reset_index(drop=True), data_normalized], axis=1)

# Tampilkan hasil normalisasi yang sudah dirapikan
print(df_normalized.head())
```
## Modeling 
### Regresi Linier
```{code-cell} python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gunakan 'xt' sebagai target
target_column = 'xt'  # Kolom target

# Pisahkan fitur (X) dan target (y)
X = df_slide_cleaned[['xt-3', 'xt-2', 'xt-1']]  # Fitur (kolom lain)
y = df_slide_cleaned[target_column]  # Kolom target ('xt')

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat model regresi linier
model_lr = LinearRegression()

# Latih model dengan data training
model_lr.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = model_lr.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
```

### Random Forest
```{code-cell} python
from sklearn.ensemble import RandomForestRegressor  # Mengimpor RandomForestRegressor dari scikit-learn.

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)  # Membuat instance RandomForestRegressor dengan 100 pohon dan seed untuk reproduksibilitas.
rf_model.fit(X_train, y_train)  # Melatih model dengan data pelatihan.

# Prediksi pada data uji
y_pred_rf = rf_model.predict(X_test)  # Menggunakan model untuk memprediksi nilai pada data uji.

# Evaluasi model
mse_rf = mean_squared_error(y_test, y_pred_rf)  # Menghitung Mean Squared Error (MSE) untuk prediksi Random Forest.
r2_rf = r2_score(y_test, y_pred_rf)  # Menghitung R-squared (R²) untuk mengevaluasi model.

# Menghitung RMSE
rmse_rf = np.sqrt(mse_rf)  # Menghitung Root Mean Squared Error (RMSE) dari MSE.

# Menampilkan hasil evaluasi
print(f'Root Mean Squared Error (Random Forest): {rmse_rf}')  # Mencetak RMSE untuk model Random Forest.
print(f'Mean Squared Error (Random Forest): {mse_rf}')  # Mencetak MSE untuk model Random Forest.
print(f'R-squared (Random Forest): {r2_rf}')  # Mencetak R² untuk model Random Forest.
```
```{code-cell} python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Data dummy untuk contoh
X = np.random.rand(10, 3)  # Fitur
y = np.random.rand(10)     # Target

# Bagi data menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Latih model
rf_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = rf_model.predict(X_test)

# Hitung MAPE secara manual
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Menghitung MAPE
mape = calculate_mape(y_test, y_pred)

print(f'MAPE: {mape:.2f}%')

# Alternatif: menggunakan sklearn untuk menghitung MAPE (jika versi sklearn mendukung)
mape_sklearn = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f'MAPE (dengan sklearn): {mape_sklearn:.2f}%')
```

### Tugas Evaluasi
<p>perbedaan pohon keputusan (decision tree) untuk regresi dengan random forest, hitung manual pohon keputusan untuk regresi untuk peramalan, membangun data time series menjadi data supervised</p>

#### Perbedaan
<p>1. Pohon Keputusan untuk Regresi
- Struktur Dasar: Pohon keputusan adalah algoritma yang membagi data ke dalam simpul-simpul berdasarkan fitur dan nilai tertentu hingga mencapai simpul daun.
- Prediksi: Untuk regresi, simpul daun berisi nilai rata-rata atau median dari target (variabel dependen) untuk data dalam simpul tersebut.
- Overfitting: Pohon keputusan tunggal cenderung overfitting, terutama jika pohon sangat dalam (tidak dipangkas).
- Kecepatan: Cepat untuk dibangun dan dijalankan, tetapi mungkin tidak cukup akurat untuk data yang kompleks.
- Kelemahan: Sensitif terhadap perubahan kecil pada data. Dataset yang sedikit berubah dapat menghasilkan pohon yang sangat berbeda (kurang stabil).
2. Random Forest untuk Regresi
- Struktur Dasar: Random forest adalah ensemble dari banyak pohon keputusan yang dibangun dengan menggunakan subset data dan fitur secara acak.
- Prediksi: Hasil regresi dihitung sebagai rata-rata prediksi dari semua pohon dalam ensemble.
- Overfitting: Random forest mengurangi overfitting dengan melakukan averaging dari banyak pohon, sehingga lebih tahan terhadap noise dalam data.
- Kecepatan: Membutuhkan lebih banyak waktu dan sumber daya dibandingkan dengan pohon keputusan karena melibatkan banyak pohon.
- Kelemahan: Random forest dapat menjadi kurang interpretatif karena terdiri dari banyak pohon (bukan model tunggal).
</p>
