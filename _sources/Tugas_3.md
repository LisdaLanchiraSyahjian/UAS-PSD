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

# PREDIKSI KEBAKARAN HUTAN
## PENDAHULUAN
### Latar Belakang
<p>Kebakaran hutan merupakan salah satu masalah lingkungan yang signifikan dan berdampak besar terhadap ekosistem, manusia, serta perubahan iklim. Di kawasan hutan Algerian, kebakaran hutan sering kali terjadi akibat kombinasi faktor alam dan aktivitas manusia, seperti suhu tinggi, kelembapan rendah, angin kencang, serta kelalaian manusia. Dampak kebakaran hutan meliputi hilangnya keanekaragaman hayati, kerusakan lingkungan, emisi gas rumah kaca, serta ancaman terhadap kehidupan masyarakat sekitar. Oleh karena itu, diperlukan upaya prediksi yang tepat untuk mengurangi risiko dan dampak kebakaran hutan melalui teknologi berbasis data.

Perkembangan teknologi data science dan machine learning memberikan peluang besar untuk memprediksi kebakaran hutan dengan memanfaatkan data historis dan parameter lingkungan, seperti suhu, kelembapan, kecepatan angin, dan curah hujan. Prediksi yang akurat dapat membantu pihak berwenang dalam mengambil tindakan pencegahan dan mitigasi yang lebih efektif.</p>

### Rumusan Masalah
<p>1. Bagaimana memanfaatkan data historis dan parameter lingkungan untuk memprediksi kebakaran hutan di hutan Algerian secara akurat?

2. Algoritma atau metode prediksi apa yang paling efektif untuk digunakan dalam memprediksi kebakaran hutan berdasarkan parameter lingkungan yang tersedia?</p>

### Tujuan
<p>1. Mengembangkan model prediksi kebakaran hutan di kawasan hutan Algerian menggunakan data historis dan parameter lingkungan.

2. Mengevaluasi performa model prediksi dengan menggunakan metode machine learning untuk memastikan tingkat akurasi yang tinggi.</p>

### Menampilkan Data
```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
```{code-cell} python
import pandas as pd

# Use the raw content link for the dataset
data_df = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/LisdaLanchiraSyahjian/Proyek-Sain-Data/main/AFF%20dataset.csv"))
data_df.head()
```

### Data Undeerstanding
<p>Pada pekerjaan kali ini, kami akan melakukan klasifikasi untuk prediksi kebakaran hutan di negara Algerian. Tujuan dari pekerjaan ini adalah untuk membantu dalam memprediksi apakah pada wilayah tersebut berpotensi terjadi kebakaran atau tidak berdasarkan berbagai fitur yang tersedia. Dataset yang kami gunakan adalah dataset "Algerian Forest Fires" yang kami ambil dari UCI Machine Learning Repository. Dataset ini berasal dari Cleveland Clinic Foundation.

Langkah pertama yang dilakukan adalah mengumpulkan data. Data tersebut berada di aiven.com, sehingga data perlu ditarik dari sumber tersebut. Dataset ini terdiri dari 244 baris dan 12 fitur yaitu Date, Temperature, RH, Ws, Rain ,FFMC, DMC, DC, ISI, BUI, FWI, Classes .</p>

```{code-cell} python
# Memeriksa apakah kolom 'Date' ada
if 'Date' not in data_df.columns:
    raise KeyError("Kolom 'Date' tidak ditemukan. Periksa nama kolom dalam dataset.")

# Mengatur kolom 'Date' sebagai indeks dan mengubah formatnya
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)

# Memastikan kolom yang diperlukan ada
required_columns = ['Temperature', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes  ']
data_df = data_df[required_columns]

# Menghapus missing values
data_df.dropna(inplace=True)
```
```{code-cell} python
data_df.plot()
```

### Mencari Missing Value
```{code-cell} python
# Mencari Missing Value
data_df.isnull().sum()
```

```{code-cell} python
import seaborn as sns
# Menghitung korelasi antar fitur untuk subset yang diinginkan
features = data_df[['Temperature', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes  ']]
correlation_matrix = features.corr()

# Menampilkan matriks korelasi
print("Matriks Korelasi:")
print(correlation_matrix)

# Menggambar heatmap untuk visualisasi korelasi
plt.figure(figsize=(10, 6))
plt.title("Heatmap Korelasi antar Fitur")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
```

```{code-cell} python
# Deskripsi Statistik
data_df.describe()
```

### Prepocessing
```{code-cell} python
# Fungsi untuk membuat sliding windows
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Menyiapkan data untuk sliding windows
window_size = 3  # Ukuran window
# Instead of using 'Close' and 'High', we will use 'Temperature' and ' RH' for prediction. You can choose other columns if needed.
data_values = data_df[['Temperature', ' RH']].values
X, y = create_sliding_windows(data_values, window_size)

# Membuat DataFrame untuk hasil sliding windows
# Adjust column names to reflect the selected features
sliding_window_df = pd.DataFrame(X.reshape(X.shape[0], -1), columns=[f'Temperature_t-{window_size-i}' for i in range(window_size)] + [f'RH_t-{window_size-i}' for i in range(window_size)])
sliding_window_df['Target_Temperature_t'] = y[:, 0]  # Target Temperature
sliding_window_df['Target_RH_t'] = y[:, 1]  # Target RH

# Menampilkan hasil sliding windows
print(sliding_window_df.head())
```

### Normalisasi
```{code-cell} python
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Close_t-3, Close_t-2, Close_t-1, High_t-3, High_t-2, High_t-1)
df_features_normalized = pd.DataFrame(
    scaler_features.fit_transform(sliding_window_df.iloc[:, :-2]),  # Ambil semua kolom kecuali target
    columns=sliding_window_df.columns[:-2],  # Nama kolom tanpa target
    index=sliding_window_df.index
)

# Normalisasi target (Target_Close_t dan Target_High_t)
df_target_normalized = pd.DataFrame(
    scaler_target.fit_transform(sliding_window_df[['Target_Temperature_t' , 'Target_RH_t']]),
    columns=['Target_Temperature_t' , 'Target_RH_t'],
    index=sliding_window_df.index
)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)

# Menampilkan hasil normalisasi
print(df_normalized.head())

# Mengatur fitur (X) dan target (y)
X = df_normalized[['Temperature_t-3' , 'Temperature_t-2' , 'Temperature_t-1' , 'RH_t-3' , 'RH_t-2' , 'RH_t-1']]
y = df_normalized[['Target_Temperature_t' , 'Target_RH_t']]  # Target adalah harga yang dinormalisasi

# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print('===== Data Train =====')
print(X_train)

print('===== Data Testing ====')
print(X_test)

# Mengambil nilai tanggal dari indeks X_train dan X_test
dates_train = X_train.index
dates_test = X_test.index

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Plot data Close dan High dengan format tanggal di sumbu x
plt.figure(figsize=(14, 7))

# Plot Close
plt.plot(data_df.index, data_df['Temperature'], label='Temperature', linestyle='-', color='blue')

# Plot High
#plt.plot(data_df.index, data_df['RH'], label='RH' , linestyle='--', color='orange')
plt.plot(sliding_window_df.index, sliding_window_df['Target_RH_t'], label='RH', color='orange', linestyle='--')

# Format sumbu x agar menampilkan tanggal
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Menampilkan label tanggal per bulan

plt.title('Temperature and RH Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)  # Putar label tanggal agar tidak tumpang tindih
plt.tight_layout()
plt.show()
```

## Pemodelan
### Regresi Linier
```{code-cell} python
# Membuat model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi harga menggunakan model
y_pred = model.predict(X_test)

# Menghitung Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Membuat plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Garis identitas
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title(f'Plot Nilai Aktual vs Prediksi\nMSE: {mse:.2f}')
plt.grid()
plt.show()
```

### Random forest
```{code-cell} python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Membuat model Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Memprediksi nilai menggunakan model
y_pred = rf_model.predict(X_test)

# Menghitung Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Membuat plot
plt.figure(figsize=(10, 6))

# Untuk y_test yang mungkin memiliki beberapa target, gunakan hanya satu kolom untuk plot
if y_test.shape[1] > 1:  # Jika y memiliki lebih dari satu target
    target_index = 0  # Pilih target ke-0 (misalnya Target_Temperature_t)
    plt.scatter(y_test.iloc[:, target_index], y_pred[:, target_index], alpha=0.5, label='Target_Temperature_t')
else:
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')

# Garis identitas
plt.plot([y_test.min().min(), y_test.max().max()], [y_test.min().min(), y_test.max().max()], 'r--', label='Identity Line')

plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title(f'Plot Nilai Aktual vs Prediksi\nMSE: {mse:.2f}')
plt.legend()
plt.grid()
plt.show()
```
