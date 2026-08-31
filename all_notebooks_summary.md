## Notebook: 1-veri_yukleme_inceleme.ipynb
### Code Cell 1
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Veri Yükleme ve İlgili Özelliği Çıkarma
# Zip'ten çıkardığın dosyanın adını doğru yazdığından emin ol!
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Tıpkı plandaki "Close" fiyatı gibi, sadece tahmin edeceğimiz sütunu çıkarıyoruz
solar_data = df[['AT_solar_generation_actual']].dropna()
print(f"Toplam Saatlik Veri Sayısı: {len(solar_data)}")

# 2. Veri Normalizasyonu (Ölçeklendirme)
# Kılavuza göre veriyi [-1, 1] aralığına çekmek için MinMaxScaler kullanıyoruz
scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

print("\nNormalize edilmiş (ölçeklendirilmiş) ilk 5 veri:")
print(solar_data_scaled[:5])

```

### Code Cell 2
```python

```


================================================================================

## Notebook: 10-veri seti analizi.ipynb
### Code Cell 1
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Yükleme
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

# 2. İstatistiksel Rapor (Matematiksel Dağılım)
print("=====================================================")
print("            VERİ SETİ İSTATİSTİKSEL ÖZETİ            ")
print("=====================================================")
# describe() fonksiyonu ortalama, standart sapma, min, max ve çeyreklik değerleri verir
print(data.describe().round(2))
print("=====================================================\n")

# 3. Görsel Dağılım Analizi (Histogram ve Kutu Grafiği)
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
feature_names = ['Güneş Enerjisi', 'Rüzgar Enerjisi', 'Elektrik Tüketimi']
colors_hist = ['#ff9999', '#66b3ff', '#99ff99']
colors_box = ['#ffcccc', '#cce5ff', '#ccffcc']

for i, col in enumerate(features):
    # Sol Sütun: Histogram (Verilerin Hangi Aralıklarda Yoğunlaştığını Gösterir)
    sns.histplot(data[col], bins=50, ax=axes[i, 0], color=colors_hist[i], kde=True)
    axes[i, 0].set_title(f'{feature_names[i]} - Dağılım Histogramı', fontweight='bold')
    axes[i, 0].set_xlabel('Megawatt (MW)')
    axes[i, 0].set_ylabel('Kayıt Sayısı (Saat)')

    # Sağ Sütun: Boxplot / Kutu Grafiği (Aykırı Değerleri / Outliers Gösterir)
    sns.boxplot(x=data[col], ax=axes[i, 1], color=colors_box[i])
    axes[i, 1].set_title(f'{feature_names[i]} - Aykırı Değer (Boxplot)', fontweight='bold')
    axes[i, 1].set_xlabel('Megawatt (MW)')

plt.tight_layout()
plt.show()
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 11-Gunluk_Net_Yuk_ve_Sebeke_Esnekligi_Tahmini.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time
import copy

# ==========================================
# 1. CİHAZ VE VERİ HAZIRLIĞI
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Aykırı Değer (Outlier) Temizliği
df.loc[df['AT_load_actual_entsoe_transparency'] < 4000, 'AT_load_actual_entsoe_transparency'] = np.nan
df['AT_load_actual_entsoe_transparency'] = df['AT_load_actual_entsoe_transparency'].interpolate(method='time')

features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# Günlük (24 Saatlik) Çoklu Tahmin İçin Kayan Pencere
def create_daily_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback), :])
        y_target = data[(i + lookback) : (i + lookback + horizon), :]
        y.append(y_target.flatten()) 
    return np.array(X), np.array(y)

lookback = 48  
horizon = 24   
X, y = create_daily_sequences(data_scaled, lookback, horizon)

train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).float()
X_test_t = torch.from_numpy(X_test).float()

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)

print("Veri Seti Hazırlandı!")

# ==========================================
# 2. MODEL MİMARİSİ (Sadece GRU - En Hızlı ve Etkili)
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

# ==========================================
# 3. ERKEN DURDURMALI EĞİTİM
# ==========================================
def train_gru(model, patience=5, epochs=30):
    print(f"\n--- GRU Modeli Eğitimi Başlıyor ---")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5) 
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Erken Durdurma! Model donduruldu (Epoch {epoch+1}).")
                break
                
    print(f"Eğitim Tamamlandı! Süre: {time.time() - start_time:.0f}s | En İyi Val Loss: {best_val_loss:.5f}")
    
    model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).cpu().numpy()
    return preds

gru_model = GRUModel(input_dim=3, hidden_dim=64, layer_dim=2, output_dim=72)
gru_preds = train_gru(gru_model)

# ==========================================
# 4. METRİKLER VE ŞEBEKE ESNEKLİĞİ (F1-SCORE)
# ==========================================
def inverse_transform_daily(data_72d):
    data_reshaped = data_72d.reshape(-1, 3)
    return scaler.inverse_transform(data_reshaped)

y_test_mw = inverse_transform_daily(y_test)
gru_preds_mw = np.clip(inverse_transform_daily(gru_preds), 0, None)

def calculate_metrics(y_true, y_pred, feature_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    wape = (mae / np.mean(y_true)) * 100 if np.mean(y_true) > 0 else 0
    print(f"[{feature_name}] -> RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")

print("\n--- GRU MODELİ 24 SAATLİK GÜNLÜK TAHMİN HATA RAPORU ---")
calculate_metrics(y_test_mw[:, 0], gru_preds_mw[:, 0], "Güneş")
calculate_metrics(y_test_mw[:, 1], gru_preds_mw[:, 1], "Rüzgar")
calculate_metrics(y_test_mw[:, 2], gru_preds_mw[:, 2], "Tüketim")

net_load_actual = y_test_mw[:, 2] - (y_test_mw[:, 0] + y_test_mw[:, 1])
net_load_gru = gru_preds_mw[:, 2] - (gru_preds_mw[:, 0] + gru_preds_mw[:, 1])

CRITICAL_THRESHOLD = 6000 
kriz_gercek = (net_load_actual > CRITICAL_THRESHOLD).astype(int)
kriz_tahmin = (net_load_gru > CRITICAL_THRESHOLD).astype(int)

print("\n=====================================================")
print("     ŞEBEKE ESNEKLİK VE KRİZ TESPİT RAPORU (GRU)     ")
print("=====================================================")
print(f"Kriz Yakalama (Recall): %{recall_score(kriz_gercek, kriz_tahmin)*100:.2f}")
print(f"Hassasiyet (Precision): %{precision_score(kriz_gercek, kriz_tahmin)*100:.2f}")
print(f"F1-Score (Başarı)     : %{f1_score(kriz_gercek, kriz_tahmin)*100:.2f}")
print("=====================================================")

# ==========================================
# 5. GÖRSELLEŞTİRME
# ==========================================
saat_sayisi = 96 
zaman_ekseni = range(saat_sayisi)
fig, axes = plt.subplots(4, 1, figsize=(16, 20))

axes[0].plot(zaman_ekseni, net_load_actual[:saat_sayisi], label='Gerçek Net Yük', color='black', linewidth=2.5)
axes[0].plot(zaman_ekseni, net_load_gru[:saat_sayisi], label='GRU Tahmini Net Yük', color='red', linestyle='--', linewidth=2)
axes[0].axhline(y=CRITICAL_THRESHOLD, color='orange', linestyle='-.', linewidth=2, label=f'Kriz Eşiği ({CRITICAL_THRESHOLD} MW)')
axes[0].fill_between(zaman_ekseni, CRITICAL_THRESHOLD, net_load_actual[:saat_sayisi], where=(net_load_actual[:saat_sayisi] > CRITICAL_THRESHOLD), color='red', alpha=0.2, label='Gerçekleşen Kriz Anları')
axes[0].set_title('Şebeke Net Yükü ve Esneklik Krizi Tespiti (İlk 4 Gün)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Megawatt (MW)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 2], label='Gerçek Tüketim', color='black', linewidth=2)
axes[1].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 2], label='GRU Tahmini', color='blue', linestyle='--')
axes[1].set_title('Elektrik Tüketimi (Şebeke Yükü)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Megawatt (MW)', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':', alpha=0.7)

axes[2].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 1], label='Gerçek Rüzgar', color='black', linewidth=2)
axes[2].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 1], label='GRU Tahmini', color='green', linestyle='--')
axes[2].set_title('Rüzgar Enerjisi Üretimi', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Megawatt (MW)', fontsize=12)
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle=':', alpha=0.7)

axes[3].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 0], label='Gerçek Güneş', color='black', linewidth=2)
axes[3].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 0], label='GRU Tahmini', color='orange', linestyle='--')
axes[3].set_title('Güneş Enerjisi Üretimi', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Zaman (Saat)', fontsize=12)
axes[3].set_ylabel('Megawatt (MW)', fontsize=12)
axes[3].legend(loc='upper right')
axes[3].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
```

### Code Cell 2
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Veriyi Okuma
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# 3. Sadece 1 Feature (Tüketim) Seçimi ve Temizlik
features_only_load = ['AT_load_actual_entsoe_transparency']
data_load = df[features_only_load].dropna()

# 4. Ölçeklendirme
scaler_load = MinMaxScaler(feature_range=(-1, 1))
data_scaled_load = scaler_load.fit_transform(data_load.values)

# 5. Kayan Pencere (Sliding Window)
def create_daily_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback), :])
        y.append(data[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled_load, 48, 24)

# 6. Veriyi Train/Validation/Test Olarak Bölme
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train, y_train = torch.from_numpy(X[:train_size]).float(), torch.from_numpy(y[:train_size]).float()
X_val, y_val = torch.from_numpy(X[train_size : train_size + val_size]).float(), torch.from_numpy(y[train_size : train_size + val_size]).float()
X_test, y_test = torch.from_numpy(X[train_size + val_size:]).float(), torch.from_numpy(y[train_size + val_size:]).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

# 7. Sadece 1 Girdi Alan GRU Modeli
class GRU_Single(nn.Module):
    def __init__(self):
        super(GRU_Single, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 24) # Çıktı sadece 24 saatlik tüketim

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

model_a = GRU_Single().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_a.parameters(), lr=0.001)

# 8. Erken Durdurmalı (Early Stopping) Eğitim Döngüsü
best_val_loss = float('inf')
best_weights = copy.deepcopy(model_a.state_dict())
patience = 5
epochs_no_improve = 0

print("--- SADECE TÜKETİM (MODEL A) EĞİTİMİ BAŞLADI ---")
for epoch in range(25):
    model_a.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model_a(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        
    model_a.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model_a(X_batch), y_batch).item()
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model_a.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Erken Durdurma (Epoch {epoch+1})")
            break

model_a.load_state_dict(best_weights)
model_a.eval()
with torch.no_grad():
    preds = model_a(X_test.to(device)).cpu().numpy()

# 9. Metrik Hesaplama (Ablation Study Sonucu)
preds_mw = scaler_load.inverse_transform(preds)
y_test_mw = scaler_load.inverse_transform(y_test.numpy())

mae = mean_absolute_error(y_test_mw, preds_mw)
wape = (mae / np.mean(y_test_mw)) * 100
print(f"\n[SADECE TÜKETİM] -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")
```

### Code Cell 3
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import copy

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Veri Okuma ve Hazırlık (3 Değişkenli Ana Formata Geri Dönüyoruz)
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

# 3. Ölçeklendirme
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# 4. Kayan Pencere
def create_daily_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback), :])
        y.append(data[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

# 5. Veri Bölme (Train / Val / Test)
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train, y_train = torch.from_numpy(X[:train_size]).float(), torch.from_numpy(y[:train_size]).float()
X_val, y_val = torch.from_numpy(X[train_size : train_size + val_size]).float(), torch.from_numpy(y[train_size : train_size + val_size]).float()
X_test, y_test = torch.from_numpy(X[train_size + val_size:]).float(), torch.from_numpy(y[train_size + val_size:]).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

# ==========================================
# 6. MİMARİ TANIMLAMALARI
# ==========================================
class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 72) # 3 değişken x 24 saat

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTM_Model(nn.Module):
    def __init__(self):
        super(BiLSTM_Model, self).__init__()
        # bidirectional=True ile model veriyi hem geçmişten geleceğe hem gelecekten geçmişe okur
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, 72) # Çift yönlü olduğu için gizli katman boyutu 2 katına (128) çıkar

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 7. ORTAK EĞİTİM VE TEST FONKSİYONU
# ==========================================
def train_and_evaluate(model, model_name, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
    print(f"\n--- {model_name} EĞİTİMİ BAŞLADI (LR: {lr}) ---")
    for epoch in range(25):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"{model_name}: Erken Durdurma Tetiklendi (Epoch {epoch+1})")
                break
                
    # En iyi ağırlıkları yükle ve test et
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    # Sadece Tüketim (Load) metriklerini hesaplamak için ters ölçeklendirme
    preds_mw = scaler.inverse_transform(preds.reshape(-1, 3)).reshape(preds.shape)
    y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, 3)).reshape(y_test.shape)

    # İndeks 2 (Load) değerlerini çekiyoruz (0: Güneş, 1: Rüzgar, 2: Load)
    preds_load = preds_mw[:, 2::3] 
    y_test_load = y_test_mw[:, 2::3]

    mae = mean_absolute_error(y_test_load, preds_load)
    wape = (mae / np.mean(y_test_load)) * 100
    
    print(f"[{model_name}] Başarıyla Tamamlandı -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")

# ==========================================
# 8. ÇALIŞTIRMA VE LİDERLİK TABLOSU OLUŞTURMA
# ==========================================
lstm_model = LSTM_Model()
bilstm_model = BiLSTM_Model()

train_and_evaluate(lstm_model, "LSTM")
train_and_evaluate(bilstm_model, "Bi-LSTM")
```

### Code Cell 4
```python

```


================================================================================

## Notebook: 12-GRU_CROSS_VALIDATION_ANALYSE.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import copy

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Veri Okuma ve Hazırlık
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

# 3. Ölçeklendirme
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# 4. Kayan Pencere
def create_daily_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback), :])
        y.append(data[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

# 5. GRU Modeli Sınıfı
class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 72)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 6. TIME-SERIES CROSS VALIDATION (ÇAPRAZ DOĞRULAMA)
# ==========================================
n_splits = 3  # Veriyi 3 farklı parçaya bölerek test edeceğiz
tscv = TimeSeriesSplit(n_splits=n_splits)

fold = 1
wape_scores = []
mae_scores = []

print(f"--- ZAMAN SERİSİ ÇAPRAZ DOĞRULAMA ({n_splits} FOLD) BAŞLIYOR ---")

for train_index, test_index in tscv.split(X):
    print(f"\n[Fold {fold}] Eğitim Seti Boyutu: {len(train_index)}, Test Seti Boyutu: {len(test_index)}")

    # Her fold içinde Train'in son %15'ini Validation (Doğrulama) için ayırıyoruz
    fold_train_size = int(len(train_index) * 0.85)
    real_train_idx = train_index[:fold_train_size]
    val_idx = train_index[fold_train_size:]
    
    X_train, y_train = torch.from_numpy(X[real_train_idx]).float(), torch.from_numpy(y[real_train_idx]).float()
    X_val, y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(y[val_idx]).float()
    X_test, y_test = torch.from_numpy(X[test_index]).float(), torch.from_numpy(y[test_index]).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    # Her fold için modeli ve optimizasyonu TAMAMEN SIFIRLIYORUZ ki önceki fold'dan ezberlemesin
    model = GRU_Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
    # Eğitim Döngüsü
    for epoch in range(25):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
                
    # En iyi ağırlıkları yükle ve test et
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    # Sadece Tüketim (Load) metriklerini hesaplamak için ters ölçeklendirme
    preds_mw = scaler.inverse_transform(preds.reshape(-1, 3)).reshape(preds.shape)
    y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, 3)).reshape(y_test.shape)

    preds_load = preds_mw[:, 2::3] 
    y_test_load = y_test_mw[:, 2::3]

    mae = mean_absolute_error(y_test_load, preds_load)
    wape = (mae / np.mean(y_test_load)) * 100
    
    print(f"Fold {fold} Tamamlandı -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")
    
    mae_scores.append(mae)
    wape_scores.append(wape)
    fold += 1

print("\n=============================================")
print(f"NİHAİ ÇAPRAZ DOĞRULAMA (CROSS-VALIDATION) SONUCU")
print(f"Ortalama MAE : {np.mean(mae_scores):.2f} MW")
print(f"Ortalama WAPE: %{np.mean(wape_scores):.2f}")
print("=============================================")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 13-zaman_kodlamasi_analizi.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import copy

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Veri Okuma ve Filtreleme (Performans uyarısını çözen adım)
print("Veri yükleniyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']

# Sadece ihtiyacımız olan sütunları alıp kopyalıyoruz (Fragmentasyon hatasını önler)
data = df_raw[base_features].copy().dropna()

# 3. Zaman Kodlaması (Time Encoding - Sin/Cos)
print("Zaman Kodlaması (Sin/Cos) uygulanıyor...")
# Yeni sütunları ayrı bir DataFrame'de toplayıp pd.concat ile birleştiriyoruz
time_features = pd.DataFrame(index=data.index)
time_features['hour'] = time_features.index.hour
time_features['month'] = time_features.index.month

time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)

# Sadece sin/cos sütunlarını ana veriye ekliyoruz
data = pd.concat([data, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

# 4. Ölçeklendirme
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# 5. Kayan Pencere
def create_daily_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

# 6. Veri Bölme (Train / Val / Test)
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train, y_train = torch.from_numpy(X[:train_size]).float(), torch.from_numpy(y[:train_size]).float()
X_val, y_val = torch.from_numpy(X[train_size : train_size + val_size]).float(), torch.from_numpy(y[train_size : train_size + val_size]).float()
X_test, y_test = torch.from_numpy(X[train_size + val_size:]).float(), torch.from_numpy(y[train_size + val_size:]).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

# ==========================================
# 7. MİMARİ TANIMLAMALARI (Girdi boyutu 7 oldu)
# ==========================================
# 3 (Güneş, Rüzgar, Tüketim) + 4 (Zaman Kodlamaları) = 7 Değişken
INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24 # Tüm 7 değişkenin 24 saatlik tahmini

class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTM_Model(nn.Module):
    def __init__(self):
        super(BiLSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_SIZE) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 8. ORTAK EĞİTİM VE TEST FONKSİYONU
# ==========================================
def train_and_evaluate(model, model_name, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
    print(f"\n--- {model_name} EĞİTİMİ BAŞLADI (Zaman Kodlamalı) ---")
    for epoch in range(25):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"{model_name}: Erken Durdurma Tetiklendi (Epoch {epoch+1})")
                break
                
    # En iyi ağırlıkları yükle ve test et
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    # Ters ölçeklendirme işlemi (7 değişken olduğu için dinamik reshape yapıyoruz)
    preds_mw = scaler.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
    y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, INPUT_SIZE)).reshape(y_test.shape)

    # İndeks 2 (Load - Tüketim) değerlerini çekiyoruz
    preds_load = preds_mw[:, 2::INPUT_SIZE] 
    y_test_load = y_test_mw[:, 2::INPUT_SIZE]

    mae = mean_absolute_error(y_test_load, preds_load)
    wape = (mae / np.mean(y_test_load)) * 100
    
    print(f"[{model_name}] Başarıyla Tamamlandı -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")

# ==========================================
# 9. ÇALIŞTIRMA VE LİDERLİK TABLOSU OLUŞTURMA
# ==========================================
gru_model = GRU_Model()
lstm_model = LSTM_Model()
bilstm_model = BiLSTM_Model()

train_and_evaluate(gru_model, "GRU")
train_and_evaluate(lstm_model, "LSTM")
train_and_evaluate(bilstm_model, "Bi-LSTM")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 14-LSTM_Cross_Validation.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Veri yükleniyor ve Zaman Kodlaması uygulanıyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df_raw[base_features].copy().dropna()

time_features = pd.DataFrame(index=data.index)
time_features['hour'] = time_features.index.hour
time_features['month'] = time_features.index.month
time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)

data = pd.concat([data, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

def create_daily_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
fold = 1
wape_scores, mae_scores = [], []

print("--- LSTM ZAMAN SERİSİ ÇAPRAZ DOĞRULAMA (3 FOLD) BAŞLIYOR ---")

for train_index, test_index in tscv.split(X):
    fold_train_size = int(len(train_index) * 0.85)
    real_train_idx = train_index[:fold_train_size]
    val_idx = train_index[fold_train_size:]
    
    X_train, y_train = torch.from_numpy(X[real_train_idx]).float(), torch.from_numpy(y[real_train_idx]).float()
    X_val, y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(y[val_idx]).float()
    X_test, y_test = torch.from_numpy(X[test_index]).float(), torch.from_numpy(y[test_index]).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    model = LSTM_Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience, epochs_no_improve = 5, 0
    
    for epoch in range(25):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    preds_mw = scaler.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
    y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, INPUT_SIZE)).reshape(y_test.shape)

    preds_load = preds_mw[:, 2::INPUT_SIZE] 
    y_test_load = y_test_mw[:, 2::INPUT_SIZE]

    mae = mean_absolute_error(y_test_load, preds_load)
    wape = (mae / np.mean(y_test_load)) * 100
    
    print(f"[Fold {fold}] Tamamlandı -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")
    mae_scores.append(mae)
    wape_scores.append(wape)
    fold += 1

print("\n=============================================")
print(f"LSTM NİHAİ ÇAPRAZ DOĞRULAMA SONUCU")
print(f"Ortalama MAE : {np.mean(mae_scores):.2f} MW")
print(f"Ortalama WAPE: %{np.mean(wape_scores):.2f}")
print("=============================================")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 15_BİLSTM_CROSS_VALIDATION.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Veri yükleniyor ve Zaman Kodlaması uygulanıyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df_raw[base_features].copy().dropna()

time_features = pd.DataFrame(index=data.index)
time_features['hour'] = time_features.index.hour
time_features['month'] = time_features.index.month
time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)

data = pd.concat([data, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

def create_daily_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24

class BiLSTM_Model(nn.Module):
    def __init__(self):
        super(BiLSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_SIZE) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
fold = 1
wape_scores, mae_scores = [], []

print("--- Bi-LSTM ZAMAN SERİSİ ÇAPRAZ DOĞRULAMA (3 FOLD) BAŞLIYOR ---")

for train_index, test_index in tscv.split(X):
    fold_train_size = int(len(train_index) * 0.85)
    real_train_idx = train_index[:fold_train_size]
    val_idx = train_index[fold_train_size:]
    
    X_train, y_train = torch.from_numpy(X[real_train_idx]).float(), torch.from_numpy(y[real_train_idx]).float()
    X_val, y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(y[val_idx]).float()
    X_test, y_test = torch.from_numpy(X[test_index]).float(), torch.from_numpy(y[test_index]).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    model = BiLSTM_Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience, epochs_no_improve = 5, 0
    
    for epoch in range(25):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        
    preds_mw = scaler.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
    y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, INPUT_SIZE)).reshape(y_test.shape)

    preds_load = preds_mw[:, 2::INPUT_SIZE] 
    y_test_load = y_test_mw[:, 2::INPUT_SIZE]

    mae = mean_absolute_error(y_test_load, preds_load)
    wape = (mae / np.mean(y_test_load)) * 100
    
    print(f"[Fold {fold}] Tamamlandı -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")
    mae_scores.append(mae)
    wape_scores.append(wape)
    fold += 1

print("\n=============================================")
print(f"Bi-LSTM NİHAİ ÇAPRAZ DOĞRULAMA SONUCU")
print(f"Ortalama MAE : {np.mean(mae_scores):.2f} MW")
print(f"Ortalama WAPE: %{np.mean(wape_scores):.2f}")
print("=============================================")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 16_net_yuk_esneklik_tahmini.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import copy

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Veri Okuma ve Zaman Kodlaması (Fragmentasyon Korumalı)
print("Veri yükleniyor ve Zaman Kodlaması uygulanıyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df_raw[base_features].copy().dropna()

time_features = pd.DataFrame(index=data.index)
time_features['hour'] = time_features.index.hour
time_features['month'] = time_features.index.month
time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)

data = pd.concat([data, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

# 3. Ölçeklendirme
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# 4. Kayan Pencere
def create_daily_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_daily_sequences(data_scaled, 48, 24)

# ==========================================
# 5. MİMARİ TANIMLAMALARI
# ==========================================
INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24 

class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTM_Model(nn.Module):
    def __init__(self):
        super(BiLSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_SIZE) 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 6. ÇAPRAZ DOĞRULAMA (CV) İLE NET YÜK EĞİTİM DÖNGÜSÜ
# ==========================================
def run_cv_net_load(ModelClass, model_name, lr=0.001):
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    net_load_wape_scores = []
    ramp_mae_scores = []
    
    print(f"\n--- {model_name} NET YÜK VE ESNEKLİK ÇAPRAZ DOĞRULAMASI BAŞLADI ---")
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        fold_train_size = int(len(train_index) * 0.85)
        real_train_idx = train_index[:fold_train_size]
        val_idx = train_index[fold_train_size:]
        
        X_train, y_train = torch.from_numpy(X[real_train_idx]).float(), torch.from_numpy(y[real_train_idx]).float()
        X_val, y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(y[val_idx]).float()
        X_test, y_test = torch.from_numpy(X[test_index]).float(), torch.from_numpy(y[test_index]).float()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
        
        # Modeli her fold'da sıfırdan başlatıyoruz
        model = ModelClass().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        # Eğitim
        for epoch in range(25):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch), y_batch).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    break
                    
        # Test Aşaması
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(device)).cpu().numpy()
            
        # Ters Ölçeklendirme
        preds_mw = scaler.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
        y_test_mw = scaler.inverse_transform(y_test.numpy().reshape(-1, INPUT_SIZE)).reshape(y_test.shape)

        actual_solar = y_test_mw[:, 0::INPUT_SIZE]
        actual_wind  = y_test_mw[:, 1::INPUT_SIZE]
        actual_load  = y_test_mw[:, 2::INPUT_SIZE]
        
        pred_solar = preds_mw[:, 0::INPUT_SIZE]
        pred_wind  = preds_mw[:, 1::INPUT_SIZE]
        pred_load  = preds_mw[:, 2::INPUT_SIZE]

        # 1. NET YÜK HESAPLAMASI
        actual_net_load = actual_load - (actual_solar + actual_wind)
        pred_net_load = pred_load - (pred_solar + pred_wind)
        
        net_load_mae = mean_absolute_error(actual_net_load, pred_net_load)
        net_load_wape = (net_load_mae / np.mean(np.abs(actual_net_load))) * 100

        # 2. ESNEKLİK (RAMPING) HESAPLAMASI
        actual_ramp = np.abs(np.diff(actual_net_load, axis=1))
        pred_ramp = np.abs(np.diff(pred_net_load, axis=1))
        ramp_mae = mean_absolute_error(actual_ramp, pred_ramp)
        
        print(f"  [Fold {fold}] Net Yük WAPE: %{net_load_wape:.2f} | Ramping MAE: {ramp_mae:.2f} MW")
        
        net_load_wape_scores.append(net_load_wape)
        ramp_mae_scores.append(ramp_mae)
        fold += 1

    print(f"[{model_name}] NİHAİ ORTALAMA -> Net Yük WAPE: %{np.mean(net_load_wape_scores):.2f} | Ramping MAE: {np.mean(ramp_mae_scores):.2f} MW")
    print("="*60)

# ==========================================
# 7. ÇALIŞTIRMA 
# ==========================================
# Modellerin sınıflarını referans olarak fonksiyona gönderiyoruz
run_cv_net_load(GRU_Model, "GRU")
run_cv_net_load(LSTM_Model, "LSTM")
run_cv_net_load(BiLSTM_Model, "Bi-LSTM")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 17_sona doğru_Mekansal_Farkındalık.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. VERİ HAZIRLIĞI VE COĞRAFİ SİMÜLASYON
# ==========================================
print("Veri yükleniyor ve hazırlıklar yapılıyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data_region_A = df_raw[base_features].copy().dropna()

def add_time_features(df):
    time_features = pd.DataFrame(index=df.index)
    time_features['hour'] = time_features.index.hour
    time_features['month'] = time_features.index.month
    time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
    time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
    time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
    time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)
    return pd.concat([df, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

data_region_A = add_time_features(data_region_A)

# Coğrafi Farklılık (Hedef Bölge) Simülasyonu
data_region_B = data_region_A.copy()
data_region_B['AT_wind_onshore_generation_actual'] *= 0.70  
data_region_B['AT_load_actual_entsoe_transparency'] = data_region_B['AT_load_actual_entsoe_transparency'] * 0.85 + np.random.normal(0, 150, len(data_region_B))

scaler_A = MinMaxScaler(feature_range=(-1, 1))
data_scaled_A = scaler_A.fit_transform(data_region_A.values)

scaler_B = MinMaxScaler(feature_range=(-1, 1))
data_scaled_B = scaler_B.fit_transform(data_region_B.values)

def create_daily_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X_A, y_A = create_daily_sequences(data_scaled_A, 48, 24)
X_B, y_B = create_daily_sequences(data_scaled_B, 48, 24)

# ==========================================
# 2. ÖZEL KAYIP FONKSİYONU (SMOOTHNESS PENALTY)
# ==========================================
class SpatialSmoothnessLoss(nn.Module):
    def __init__(self, lambda_smooth=0.05):
        super(SpatialSmoothnessLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth # Fiziksel esneklik ceza katsayısı

    def forward(self, predictions, targets):
        base_loss = self.mse(predictions, targets)
        
        preds_reshaped = predictions.view(-1, 24, 7)
        diffs = preds_reshaped[:, 1:, :] - preds_reshaped[:, :-1, :]
        smoothness_penalty = torch.mean(diffs ** 2)
        
        return base_loss + (self.lambda_smooth * smoothness_penalty)

# ==========================================
# 3. STANDART MİMARİLER 
# ==========================================
INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24 

class Model_GRU(nn.Module):
    def __init__(self):
        super(Model_GRU, self).__init__()
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class Model_LSTM(nn.Module):
    def __init__(self):
        super(Model_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class Model_BiLSTM(nn.Module):
    def __init__(self):
        super(Model_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_SIZE) 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 4. ÇAPRAZ DOĞRULAMA (CV) + MEKANSAL TEST DÖNGÜSÜ
# ==========================================
def run_spatial_cv(ModelClass, model_name, use_smoothness_loss):
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    wape_scores = []
    mae_scores = []
    rmse_scores = []
    ramp_scores = []
    
    loss_type = "Fiziksel Pürüzsüzlük (Smoothness Penalty)" if use_smoothness_loss else "Standart Hata (MSE)"
    print(f"\n--- {model_name} | {loss_type} | ÇAPRAZ DOĞRULAMA ---")
    
    fold = 1
    for train_index, test_index in tscv.split(X_A):
        fold_train_size = int(len(train_index) * 0.85)
        real_train_idx = train_index[:fold_train_size]
        val_idx = train_index[fold_train_size:]
        
        X_train_A = torch.from_numpy(X_A[real_train_idx]).float()
        y_train_A = torch.from_numpy(y_A[real_train_idx]).float()
        X_val_A = torch.from_numpy(X_A[val_idx]).float()
        y_val_A = torch.from_numpy(y_A[val_idx]).float()

        # TEST: BÖLGE B (Kırsal / Farklı Coğrafya)
        X_test_B = torch.from_numpy(X_B[test_index]).float()
        y_test_B = torch.from_numpy(y_B[test_index]).float()

        train_loader_A = DataLoader(TensorDataset(X_train_A, y_train_A), batch_size=64, shuffle=False)
        val_loader_A = DataLoader(TensorDataset(X_val_A, y_val_A), batch_size=64, shuffle=False)
        
        model = ModelClass().to(device)
        
        if use_smoothness_loss:
            criterion = SpatialSmoothnessLoss(lambda_smooth=0.05)
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(25):
            model.train()
            for X_batch, y_batch in train_loader_A:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader_A:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch), y_batch).item()
            val_loss /= len(val_loader_A)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    break
                    
        # TEST AŞAMASI 
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            preds = model(X_test_B.to(device)).cpu().numpy()
            
        preds_mw = scaler_B.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
        y_test_B_mw = scaler_B.inverse_transform(y_test_B.numpy().reshape(-1, INPUT_SIZE)).reshape(y_test_B.shape)

        actual_solar = y_test_B_mw[:, 0::INPUT_SIZE]
        actual_wind  = y_test_B_mw[:, 1::INPUT_SIZE]
        actual_load  = y_test_B_mw[:, 2::INPUT_SIZE]
        
        pred_solar = preds_mw[:, 0::INPUT_SIZE]
        pred_wind  = preds_mw[:, 1::INPUT_SIZE]
        pred_load  = preds_mw[:, 2::INPUT_SIZE]

        # NET YÜK HESAPLAMASI
        actual_net_load = actual_load - (actual_solar + actual_wind)
        pred_net_load = pred_load - (pred_solar + pred_wind)
        
        net_load_mae = mean_absolute_error(actual_net_load, pred_net_load)
        net_load_mse = mean_squared_error(actual_net_load, pred_net_load)
        net_load_rmse = np.sqrt(net_load_mse)
        net_load_wape = (net_load_mae / np.mean(np.abs(actual_net_load))) * 100

        # ESNEKLİK (RAMPING) HESAPLAMASI
        actual_ramp = np.abs(np.diff(actual_net_load, axis=1))
        pred_ramp = np.abs(np.diff(pred_net_load, axis=1))
        ramp_mae = mean_absolute_error(actual_ramp, pred_ramp)
        
        print(f"  [Fold {fold}] WAPE: %{net_load_wape:.2f} | MAE: {net_load_mae:.2f} MW | RMSE: {net_load_rmse:.2f} MW | Ramping: {ramp_mae:.2f} MW")
        
        wape_scores.append(net_load_wape)
        mae_scores.append(net_load_mae)
        rmse_scores.append(net_load_rmse)
        ramp_scores.append(ramp_mae)
        
        fold += 1

    print(f"[{model_name}] NİHAİ ORTALAMALAR:")
    print(f" -> WAPE: %{np.mean(wape_scores):.2f}")
    print(f" -> MAE:  {np.mean(mae_scores):.2f} MW")
    print(f" -> RMSE: {np.mean(rmse_scores):.2f} MW")
    print(f" -> Ramping MAE: {np.mean(ramp_scores):.2f} MW")
    print("="*65)

# ==========================================
# 5. DENEYİ BAŞLAT
# ==========================================
print("\n=================================================================")
print("MEKANSAL FARKINDALIK + FİZİKSEL PÜRÜZSÜZLÜK TESTİ BAŞLIYOR")
print("=================================================================")

# GRU
run_spatial_cv(Model_GRU, "GRU", use_smoothness_loss=False)
run_spatial_cv(Model_GRU, "GRU", use_smoothness_loss=True)

# LSTM
run_spatial_cv(Model_LSTM, "LSTM", use_smoothness_loss=False)
run_spatial_cv(Model_LSTM, "LSTM", use_smoothness_loss=True)

# Bi-LSTM
run_spatial_cv(Model_BiLSTM, "Bi-LSTM", use_smoothness_loss=False)
run_spatial_cv(Model_BiLSTM, "Bi-LSTM", use_smoothness_loss=True)
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 18_forecast.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. VERİ HAZIRLIĞI VE KAYAN PENCERE (SLIDING WINDOW)
# ==========================================
print("Veri yükleniyor ve Kayan Pencere (Sliding Window) oluşturuluyor...")
df_raw = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
base_features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df_raw[base_features].copy().dropna()

time_features = pd.DataFrame(index=data.index)
time_features['hour'] = time_features.index.hour
time_features['month'] = time_features.index.month
time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24.0)
time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24.0)
time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12.0)
time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12.0)
data = pd.concat([data, time_features[['hour_sin', 'hour_cos', 'month_sin', 'month_cos']]], axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

def create_sequences(data_array, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data_array) - lookback - horizon + 1):
        X.append(data_array[i : (i + lookback), :])
        y.append(data_array[(i + lookback) : (i + lookback + horizon), :].flatten()) 
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 48, 24)

# Eğitim Seti 
X_train = torch.from_numpy(X[:-3]).float()
y_train = torch.from_numpy(y[:-3]).float()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)

# Forecast Seti (Geleceği tahmin etmek için ayrılan son 3 kayan pencere)
X_forecast = torch.from_numpy(X[-3:]).float()
y_actual = y[-3:] 
forecast_dates = data.index[-24 * 3:] 

# ==========================================
# 2. MİMARİLER VE PÜRÜZSÜZLÜK FONKSİYONU
# ==========================================
INPUT_SIZE = 7
OUTPUT_SIZE = 7 * 24 

class SpatialSmoothnessLoss(nn.Module):
    def __init__(self, lambda_smooth=0.05):
        super(SpatialSmoothnessLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
    def forward(self, predictions, targets):
        base_loss = self.mse(predictions, targets)
        preds_reshaped = predictions.view(-1, 24, 7)
        diffs = preds_reshaped[:, 1:, :] - preds_reshaped[:, :-1, :]
        return base_loss + (self.lambda_smooth * torch.mean(diffs ** 2))

class Model_GRU(nn.Module):
    def __init__(self):
        super(Model_GRU, self).__init__()
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class Model_LSTM(nn.Module):
    def __init__(self):
        super(Model_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, OUTPUT_SIZE)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class Model_BiLSTM(nn.Module):
    def __init__(self):
        super(Model_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_SIZE) 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# 3. EĞİTİM, ÇIKTI (PRINT) VE FORECAST DÖNGÜSÜ
# ==========================================
def train_and_forecast_with_metrics(ModelClass, model_name):
    print(f"\n--- {model_name} EĞİTİLİYOR VE FORECAST ALINIYOR ---")
    model = ModelClass().to(device)
    criterion = SpatialSmoothnessLoss(lambda_smooth=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(20):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
    # Forecast Aşaması
    model.eval()
    with torch.no_grad():
        preds = model(X_forecast.to(device)).cpu().numpy()
        
    # Ters Ölçeklendirme ve Metrik Hesaplama
    preds_mw = scaler.inverse_transform(preds.reshape(-1, INPUT_SIZE)).reshape(preds.shape)
    y_actual_mw = scaler.inverse_transform(y_actual.reshape(-1, INPUT_SIZE)).reshape(y_actual.shape)

    actual_solar = y_actual_mw[:, 0::INPUT_SIZE]
    actual_wind  = y_actual_mw[:, 1::INPUT_SIZE]
    actual_load  = y_actual_mw[:, 2::INPUT_SIZE]
    
    pred_solar = preds_mw[:, 0::INPUT_SIZE]
    pred_wind  = preds_mw[:, 1::INPUT_SIZE]
    pred_load  = preds_mw[:, 2::INPUT_SIZE]

    # NET YÜK HESAPLAMASI
    actual_net_load = actual_load - (actual_solar + actual_wind)
    pred_net_load = pred_load - (pred_solar + pred_wind)
    
    net_load_mae = mean_absolute_error(actual_net_load, pred_net_load)
    net_load_mse = mean_squared_error(actual_net_load, pred_net_load)
    net_load_rmse = np.sqrt(net_load_mse)
    net_load_wape = (net_load_mae / np.mean(np.abs(actual_net_load))) * 100

    # ESNEKLİK (RAMPING) HESAPLAMASI
    actual_ramp = np.abs(np.diff(actual_net_load, axis=1))
    pred_ramp = np.abs(np.diff(pred_net_load, axis=1))
    ramp_mae = mean_absolute_error(actual_ramp, pred_ramp)
    
    # EKRANA YAZDIRMA KISMI 
    print(f"[{model_name}] FORECAST (GELECEK TAHMİNİ) SONUÇLARI:")
    print(f" -> Net Yük WAPE: %{net_load_wape:.2f} | Net Yük MAE: {net_load_mae:.2f} MW")
    print(f" -> Net Yük MSE:  {net_load_mse:.2f} | Net Yük RMSE: {net_load_rmse:.2f} MW")
    print(f" -> Esneklik (Ramping) MAE: {ramp_mae:.2f} MW")
    
    return pred_load.flatten()

# ==========================================
# 4. POWER BI İÇİN VERİ ÇERÇEVESİ OLUŞTURMA
# ==========================================
print("\n=======================================================")
print("FORECAST TESTİ BAŞLIYOR (Çıktılar Ekrana Basılacak)")
print("=======================================================")

y_actual_full_mw = scaler.inverse_transform(y_actual.reshape(-1, INPUT_SIZE)).reshape(y_actual.shape)
actual_load_full = y_actual_full_mw[:, 2::INPUT_SIZE].flatten()

df_results = pd.DataFrame({
    'Tarih': forecast_dates,
    'Gercek_Tuketim_MW': actual_load_full,
    'GRU_Tahmin_MW': train_and_forecast_with_metrics(Model_GRU, "GRU"),
    'LSTM_Tahmin_MW': train_and_forecast_with_metrics(Model_LSTM, "LSTM"),
    'BiLSTM_Tahmin_MW': train_and_forecast_with_metrics(Model_BiLSTM, "Bi-LSTM")
})

# Hata Oranlarını Tabloya Ekleme
df_results['GRU_Hata'] = df_results['Gercek_Tuketim_MW'] - df_results['GRU_Tahmin_MW']
df_results['LSTM_Hata'] = df_results['Gercek_Tuketim_MW'] - df_results['LSTM_Tahmin_MW']
df_results['BiLSTM_Hata'] = df_results['Gercek_Tuketim_MW'] - df_results['BiLSTM_Tahmin_MW']

df_results.to_csv('power_bi_forecast_sonuclari.csv', index=False)
print("\n=======================================================")
print("Tüm modeller başarıyla test edildi! 'power_bi_forecast_sonuclari.csv' dosyası oluşturuldu.")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 2-test setleri oluşturma.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# 1. Veri Yükleme ve Temizleme
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

# 2. Normalizasyon (MinMaxScaler)
# Kılavuza göre veriyi [-1, 1] aralığına çekiyoruz
scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

# 3. Kayan Pencere (Sliding Window) Fonksiyonu
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Saatlik veri kullandığımız için 24 saatlik (1 günlük) bir pencere belirliyoruz
lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

# 4. Eğitim ve Test Kümelerine Ayırma
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Tensörlere Dönüştürme
# PyTorch modelleri tensor girdileri gerektirir
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print("Tüm Veri Hazırlığı Tek Seferde Tamamlandı!\n")
print(f"Eğitim Verisi (X_train) Tensör Boyutu: {X_train_tensor.shape}")
print(f"Test Verisi (X_test) Tensör Boyutu: {X_test_tensor.shape}")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 3-model_olusturma.ipynb
### Code Cell 1
```python
import torch
import torch.nn as nn

# 1. LSTM Model Mimarisi
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # LSTM katmanı: Giriş boyutu 1, gizli katman boyutu 32, katman sayısı 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Tam bağlı (Fully Connected) çıkış katmanı
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Başlangıç gizli (hidden) ve hücre (cell) durumlarını sıfır olarak başlatıyoruz
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM ileri beslemesi
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Sadece zaman adımının son çıktısını alarak tahmin yapıyoruz
        out = self.fc(out[:, -1, :])
        return out

# 2. GRU Model Mimarisi
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # GRU katmanı
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Çıkış katmanı
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Başlangıç gizli durumunu sıfır olarak başlatıyoruz
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU ileri beslemesi
        out, hn = self.gru(x, h0)
        
        # Son zaman adımının çıktısını alıyoruz
        out = self.fc(out[:, -1, :])
        return out

# Hiperparametreleri tanımlıyoruz
input_dim = 1     # Tek bir özellik (Güneş enerjisi üretimi)
hidden_dim = 32   # Gizli katman birim sayısı
layer_dim = 2     # Model katman sayısı
output_dim = 1    # Tahmin edilecek tek değer (Sonraki saatin üretimi)

# Cihaz belirleme (GPU varsa RTX 3050'yi kullanıyoruz)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelleri oluşturup cihaza taşıyoruz
lstm_model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
gru_model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

print("LSTM Model Mimarisi:\n", lstm_model)
print("\nGRU Model Mimarisi:\n", gru_model)
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 4-ilk_tahminler.ipynb
### Code Cell 1
```python
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Eğitim ve Test Verilerini DataLoader ile Paketleme (Batch'lere Ayırma)
# GPU'nun veriyi parça parça işleyebilmesi için DataLoader kullanıyoruz
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False) # Zaman serilerinde shuffle=False olur!

# 2. Kayıp Fonksiyonu ve Optimizasyon
criterion = nn.MSELoss()  # Regresyon hatası için MSE (Mean Squared Error)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

# 3. LSTM Modelini Eğitme (Training Loop)
print("LSTM Modeli Eğitimi Başlıyor...")
epochs = 5  # İlk test için 5 epoch veriyoruz (dilerseniz artırabilirsiniz)
lstm_model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # İleri Besleme (Forward pass)
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Geri Yayılım ve Ağırlık Güncelleme
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - Ortalama Eğitim Kaybı (Loss): {total_loss/len(train_loader):.4f}")

# 4. Test Verisi Üzerinden İlk Tahminleri (Predictions) Yapma
print("\nTest Verisi Üzerinden Tahminler Yapılıyor...")
lstm_model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    predictions = lstm_model(X_test_device)
    
print("Tahmin İşlemi Tamamlandı!")
print(f"Tahmin Edilen Çıktı Boyutu: {predictions.shape}")
print("İlk 5 Tahmin Değeri:\n", predictions[:5].cpu().numpy())
```

### Code Cell 2
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 1. Veri Yükleme ve Ön İşleme
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

# 2. Kayan Pencere (Sliding Window)
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Tensör Dönüşümleri
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# 4. Model Mimarileri (LSTM ve GRU)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model = LSTMModel(input_dim=1, hidden_dim=32, layer_dim=2, output_dim=1).to(device)

# 5. Eğitim (Training Loop)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

criterion = nn.MSELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

print("LSTM Modeli Eğitimi Başlıyor...")
epochs = 5
lstm_model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - Ortalama Eğitim Kaybı (Loss): {total_loss/len(train_loader):.4f}")

# 6. Tahmin Yapma
print("\nTest Verisi Üzerinden Tahminler Yapılıyor...")
lstm_model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    predictions = lstm_model(X_test_device)
    
print("Tahmin İşlemi Tamamlandı!")
print(f"Tahmin Edilen Çıktı Boyutu: {predictions.shape}")
print("İlk 5 Tahmin Değeri:\n", predictions[:5].cpu().numpy())
```

### Code Cell 3
```python

```


================================================================================

## Notebook: 5-LSTM_GRU_Hata_Skoru_Karsılastırma.ipynb
### Code Cell 1
```python
import time

# 1. GRU Model Mimarisini Tanımlama
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

gru_model = GRUModel(input_dim=1, hidden_dim=32, layer_dim=2, output_dim=1).to(device)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=0.001)

# 2. GRU Modelini Eğitme (Training Loop) ve Süre Ölçümü
print("GRU Modeli Eğitimi Başlıyor...")
start_time = time.time()
gru_model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = gru_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer_gru.zero_grad()
        loss.backward()
        optimizer_gru.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - GRU Eğitim Kaybı (Loss): {total_loss/len(train_loader):.4f}")

gru_training_time = time.time() - start_time
print(f"GRU Eğitim Süresi: {gru_training_time:.2f} saniye")

# 3. GRU Tahminleri
gru_model.eval()
with torch.no_grad():
    gru_predictions = gru_model(X_test_device)

# 4. Performans Karşılaştırması (MSE Metriği)
mse_criterion = nn.MSELoss()
lstm_loss = mse_criterion(predictions, y_test_tensor.to(device)).item()
gru_loss = mse_criterion(gru_predictions, y_test_tensor.to(device)).item()

print("\n--- MODEL KARŞILAŞTIRMA RAPORU ---")
print(f"LSTM Test MSE (Hata) Skoru: {lstm_loss:.6f}")
print(f"GRU Test MSE (Hata) Skoru:  {gru_loss:.6f}")
```

### Code Cell 2
```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

# 1. GRU Model Mimarisini Tanımlama
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

gru_model = GRUModel(input_dim=1, hidden_dim=32, layer_dim=2, output_dim=1).to(device)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=0.001)

# 2. GRU Modelini Eğitme (Training Loop) ve Süre Ölçümü
print("GRU Modeli Eğitimi Başlıyor...")
start_time = time.time()
gru_model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = gru_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer_gru.zero_grad()
        loss.backward()
        optimizer_gru.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - GRU Eğitim Kaybı (Loss): {total_loss/len(train_loader):.4f}")

gru_training_time = time.time() - start_time
print(f"GRU Eğitim Süresi: {gru_training_time:.2f} saniye")

# 3. GRU Tahminleri
gru_model.eval()
with torch.no_grad():
    gru_predictions = gru_model(X_test_device)

# 4. Performans Karşılaştırması (MSE Metriği)
mse_criterion = nn.MSELoss()
lstm_loss = mse_criterion(predictions, y_test_tensor.to(device)).item()
gru_loss = mse_criterion(gru_predictions, y_test_tensor.to(device)).item()

print("\n--- MODEL KARŞILAŞTIRMA RAPORU ---")
print(f"LSTM Test MSE (Hata) Skoru: {lstm_loss:.6f}")
print(f"GRU Test MSE (Hata) Skoru:  {gru_loss:.6f}")
```

### Code Cell 3
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time

# 1. Cihaz Belirleme
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan Hesaplama Cihazı: {device}")

# 2. Veri Yükleme ve Ön İşleme
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

# 3. Kayan Pencere (Sliding Window)
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Tensör Dönüşümleri ve DataLoader
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
X_test_device = X_test_tensor.to(device)
y_test_device = y_test_tensor.to(device)

criterion = nn.MSELoss()
epochs = 5

# ==========================================
# 5. LSTM MODELİ EĞİTİMİ VE TAHMİNİ
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel(input_dim=1, hidden_dim=32, layer_dim=2, output_dim=1).to(device)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

print("\n--- LSTM Modeli Eğitimi Başlıyor ---")
lstm_model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - LSTM Loss: {total_loss/len(train_loader):.4f}")

lstm_model.eval()
with torch.no_grad():
    lstm_predictions = lstm_model(X_test_device)

# ==========================================
# 6. GRU MODELİ EĞİTİMİ VE TAHMİNİ
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

gru_model = GRUModel(input_dim=1, hidden_dim=32, layer_dim=2, output_dim=1).to(device)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=0.001)

print("\n--- GRU Modeli Eğitimi Başlıyor ---")
gru_model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = gru_model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer_gru.zero_grad()
        loss.backward()
        optimizer_gru.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - GRU Loss: {total_loss/len(train_loader):.4f}")

gru_model.eval()
with torch.no_grad():
    gru_predictions = gru_model(X_test_device)

# ==========================================
# 7. PERFORMANS KARŞILAŞTIRMA RAPORU
# ==========================================
lstm_loss = criterion(lstm_predictions, y_test_device).item()
gru_loss = criterion(gru_predictions, y_test_device).item()

print("\n========================================")
print("       MODEL KARŞILAŞTIRMA RAPORU       ")
print("========================================")
print(f"LSTM Test MSE (Hata) Skoru: {lstm_loss:.6f}")
print(f"GRU Test MSE (Hata) Skoru:  {gru_loss:.6f}")
print("========================================")
```

### Code Cell 4
```python

```


================================================================================

## Notebook: 6-Watt_Cinsi_Karsılastırma.ipynb
### Code Cell 1
```python
# 1. Modelin ürettiği tahminleri ve gerçek değerleri CPU'ya alıp numpy'a çeviriyoruz
preds_lstm = lstm_predictions.cpu().numpy()
preds_gru = gru_predictions.cpu().numpy()
gercek_degerler = y_test_tensor.numpy()

# 2. MinMaxScaler kullanarak bu tahminleri -1, 1 aralığından orijinal Megawatt değerlerine geri çeviriyoruz
# Not: scaler nesnesini daha önce tanımlamıştık, doğrudan inverse_transform yapıyoruz
gercek_mw = scaler.inverse_transform(gercek_degerler)
lstm_mw = scaler.inverse_transform(preds_lstm)
gru_mw = scaler.inverse_transform(preds_gru)

# 3. İlk 10 örnek için Karşılaştırma Tablosu Oluşturma
print("---------------------------------------------------------")
print(f"{'Örnek':<8} | {'Gerçek Üretim (MW)':<20} | {'LSTM Tahmini (MW)':<18} | {'GRU Tahmini (MW)'}")
print("---------------------------------------------------------")

for i in range(10):
    print(f"{i+1:<8} | {gercek_mw[i][0]:<20.2f} | {lstm_mw[i][0]:<18.2f} | {gru_mw[i][0]:.2f}")

print("---------------------------------------------------------")
```

### Code Cell 2
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 1. Cihaz ve Veri Hazırlığı
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
X_test_device = X_test_tensor.to(device)

criterion = nn.MSELoss()
epochs = 3  # Hızlı sonuç için 3 epoch tutuyoruz

# 2. LSTM Modeli Eğitimi ve Tahmin
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel(1, 32, 2, 1).to(device)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

lstm_model.train()
for epoch in range(epochs):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        loss = criterion(lstm_model(bx), by)
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()

lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X_test_device).cpu().numpy()

# 3. GRU Modeli Eğitimi ve Tahmin
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

gru_model = GRUModel(1, 32, 2, 1).to(device)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=0.001)

gru_model.train()
for epoch in range(epochs):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        loss = criterion(gru_model(bx), by)
        optimizer_gru.zero_grad()
        loss.backward()
        optimizer_gru.step()

gru_model.eval()
with torch.no_grad():
    gru_preds = gru_model(X_test_device).cpu().numpy()

# 4. Megawatt (MW) Cinsine Çevirme ve Tablo Oluşturma
gercek_mw = scaler.inverse_transform(y_test_tensor.numpy())
lstm_mw = scaler.inverse_transform(lstm_preds)
gru_mw = scaler.inverse_transform(gru_preds)

print("---------------------------------------------------------")
print(f"{'Örnek':<8} | {'Gerçek Üretim (MW)':<20} | {'LSTM Tahmini (MW)':<18} | {'GRU Tahmini (MW)'}")
print("---------------------------------------------------------")

for i in range(10):
    print(f"{i+1:<8} | {gercek_mw[i][0]:<20.2f} | {lstm_mw[i][0]:<18.2f} | {gru_mw[i][0]:.2f}")

print("---------------------------------------------------------")
```

### Code Cell 3
```python

```


================================================================================

## Notebook: 7-Kapsamlı_model_karşılaştırma.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Cihaz ve Veri Hazırlığı
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
X_test_device = X_test_tensor.to(device)

criterion = nn.MSELoss()
epochs = 3

# 2. Modeller (LSTM, GRU, Bi-LSTM)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def train_and_predict(model_class, name):
    print(f"--- {name} Modeli Eğitiliyor ---")
    model = model_class(1, 32, 2, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test_device).cpu().numpy()
    return preds

lstm_preds = train_and_predict(LSTMModel, "LSTM")
gru_preds = train_and_predict(GRUModel, "GRU")
bilstm_preds = train_and_predict(BiLSTMModel, "Bi-LSTM")

# 3. MW Cinsine Çevirme ve Negatif Değerleri Temizleme (Clip)
gercek_mw = scaler.inverse_transform(y_test_tensor.numpy())
lstm_mw = np.clip(scaler.inverse_transform(lstm_preds), 0, None)
gru_mw = np.clip(scaler.inverse_transform(gru_preds), 0, None)
bilstm_mw = np.clip(scaler.inverse_transform(bilstm_preds), 0, None)

# 4. Kapsamlı Metrik Hesaplama (MSE, RMSE, MAE, MAPE)
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Sıfıra bölünme hatasını önlemek için 0 olan gerçek değerleri filtreliyoruz
    mask = y_true.flatten() > 1.0 
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mse, rmse, mae, mape

m_lstm = calculate_metrics(gercek_mw, lstm_mw)
m_gru = calculate_metrics(gercek_mw, gru_mw)
m_bilstm = calculate_metrics(gercek_mw, bilstm_mw)

print("\n==========================================================================")
print("             KAPSAMLI MODEL KARŞILAŞTIRMA RAPORU (MSE, RMSE, MAE, MAPE)   ")
print("==========================================================================")
print(f"LSTM   -> MSE: {m_lstm[0]:.2f} | RMSE: {m_lstm[1]:.2f} MW | MAE: {m_lstm[2]:.2f} MW | MAPE: %{m_lstm[3]:.2f}")
print(f"GRU    -> MSE: {m_gru[0]:.2f} | RMSE: {m_gru[1]:.2f} MW | MAE: {m_gru[2]:.2f} MW | MAPE: %{m_gru[3]:.2f}")
print(f"Bi-LSTM-> MSE: {m_bilstm[0]:.2f} | RMSE: {m_bilstm[1]:.2f} MW | MAE: {m_bilstm[2]:.2f} MW | MAPE: %{m_bilstm[3]:.2f}")
print("==========================================================================")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 8-Avusturya_final_gorsellestirme.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt # Görselleştirme kütüphanemiz

# 1. Cihaz ve Veri Hazırlığı
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
solar_data = df[['AT_solar_generation_actual']].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
solar_data_scaled = scaler.fit_transform(solar_data.values)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 24 
X, y = create_sequences(solar_data_scaled, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
X_test_device = X_test_tensor.to(device)

criterion = nn.MSELoss()
epochs = 20  # Modelin daha iyi öğrenmesi için eğitim döngüsünü artırdık

# 2. İyileştirilmiş Modeller (Dropout Eklendi)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        # Dropout eklendi
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def train_and_predict(model_class, name):
    print(f"--- {name} Modeli Eğitiliyor ({epochs} Epoch) ---")
    model = model_class(1, 32, 2, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test_device).cpu().numpy()
    return preds

# Modelleri Eğitme (Epoch arttığı için bu kısım biraz vakit alabilir)
lstm_preds = train_and_predict(LSTMModel, "LSTM")
gru_preds = train_and_predict(GRUModel, "GRU")
bilstm_preds = train_and_predict(BiLSTMModel, "Bi-LSTM")

# 3. MW Cinsine Çevirme ve Negatifleri Sabitleme
gercek_mw = scaler.inverse_transform(y_test_tensor.numpy())
lstm_mw = np.clip(scaler.inverse_transform(lstm_preds), 0, None)
gru_mw = np.clip(scaler.inverse_transform(gru_preds), 0, None)
bilstm_mw = np.clip(scaler.inverse_transform(bilstm_preds), 0, None)

# 4. Veriyi Görselleştirme (Matplotlib)
plt.figure(figsize=(16, 6))

# Test setindeki ilk 100 saati (yaklaşık 4 gün) çizdiriyoruz
plt.plot(gercek_mw[:100], label='Gerçek Üretim (MW)', color='black', linewidth=3)
plt.plot(lstm_mw[:100], label='LSTM Tahmini', linestyle='--', alpha=0.8)
plt.plot(gru_mw[:100], label='GRU Tahmini', linestyle='-.', alpha=0.8)
plt.plot(bilstm_mw[:100], label='Bi-LSTM Tahmini', linestyle=':', linewidth=2, alpha=0.8)

plt.title('Avusturya Güneş Enerjisi Üretimi: Gerçek vs. Yapay Zeka Tahminleri (İlk 100 Saat)', fontsize=14)
plt.xlabel('Zaman (Saat)', fontsize=12)
plt.ylabel('Enerji Üretimi (Megawatt)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

print("\nModel eğitimleri tamamlandı, grafik oluşturuluyor...")
plt.show()
```

### Code Cell 2
```python

```


================================================================================

## Notebook: 9-avusturya güneş rüzgar elektrik tüketim mae wape.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Cihaz ve Veri Hazırlığı
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Artık 3 farklı değişkenimiz (feature) var
features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

# Tüketim verisindeki 4000 MW altındaki hatalı okumaları NaN yapıyoruz
data.loc[data['AT_load_actual_entsoe_transparency'] < 4000, 'AT_load_actual_entsoe_transparency'] = np.nan

# Boşlukları zaman akışına göre doğal bir şekilde dolduruyoruz
data['AT_load_actual_entsoe_transparency'] = data['AT_load_actual_entsoe_transparency'].interpolate(method='time')

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 24)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=64, shuffle=False)
X_test_device = torch.from_numpy(X_test).float().to(device)

# 2. Tek Çatıda Birleştirilmiş Çok Değişkenli Model
class UnifiedMultiModel(nn.Module):
    def __init__(self, rnn_type):
        super(UnifiedMultiModel, self).__init__()
        bidir = (rnn_type == 'Bi-LSTM')
        rnn_class = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        # input_dim=3 oldu, gizli katmanları güçlendirdik
        self.rnn = rnn_class(3, 64, 2, batch_first=True, dropout=0.2, bidirectional=bidir)
        self.fc = nn.Linear(128 if bidir else 64, 3) # output_dim=3

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

gercek = scaler.inverse_transform(y_test)
predictions = {}
epochs = 10 # 3 değişken için ideal süre

print("--- Çok Değişkenli Modeller Eğitiliyor ---")
for name in ['LSTM', 'GRU', 'Bi-LSTM']:
    print(f"{name} eğitimi başladı...")
    model = UnifiedMultiModel(name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test_device).cpu().numpy()
    # Eksi değerleri 0'a sabitleyerek Megawatt'a dönüyoruz
    predictions[name] = np.clip(scaler.inverse_transform(preds), 0, None)

# 3. Kapsamlı WAPE Metrik Raporu
def print_metrics(y_t, y_p, feature_idx, feature_name):
    yt, yp = y_t[:, feature_idx], y_p[:, feature_idx]
    mae = mean_absolute_error(yt, yp)
    wape = np.sum(np.abs(yt - yp)) / np.sum(np.abs(yt)) * 100
    print(f"{feature_name} -> MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")

print("\n--- MODEL KARŞILAŞTIRMA RAPORU ---")
for name in ['LSTM', 'GRU', 'Bi-LSTM']:
    print(f"\n[{name} SONUÇLARI]")
    print_metrics(gercek, predictions[name], 0, "Güneş  ")
    print_metrics(gercek, predictions[name], 1, "Rüzgar ")
    print_metrics(gercek, predictions[name], 2, "Tüketim")

# 4. Üçlü Görselleştirme (Her Biri İçin Ayrı Grafik)
titles = ['Güneş Enerjisi Üretimi', 'Rüzgar Enerjisi Üretimi', 'Elektrik Tüketimi (Şebeke Yükü)']
y_labels = ['Güneş (MW)', 'Rüzgar (MW)', 'Tüketim (MW)']

for i in range(3):
    plt.figure(figsize=(14, 5)) # Her döngüde yepyeni ve geniş bir grafik oluşturur
    
    # Gerçek veri ve 3 modelin tahminleri
    plt.plot(gercek[:100, i], label='Gerçek', color='black', linewidth=3)
    plt.plot(predictions['LSTM'][:100, i], label='LSTM', linestyle='--', alpha=0.8)
    plt.plot(predictions['GRU'][:100, i], label='GRU', linestyle='-.', alpha=0.8)
    plt.plot(predictions['Bi-LSTM'][:100, i], label='Bi-LSTM', linestyle=':', linewidth=2, alpha=0.8)
    
    # Grafik ayarları
    plt.title(f'Avusturya {titles[i]} Tahmini (İlk 100 Saat)', fontsize=14, fontweight='bold')
    plt.xlabel('Zaman (Saat)', fontsize=12)
    plt.ylabel(y_labels[i], fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Her grafiği ayrı ayrı ekrana bas
    plt.show()
```

### Code Cell 2
```python

```


================================================================================

## Notebook: Untitled.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler  # İŞTE EKSİK OLAN KÜTÜPHANE!
import time
import copy

# ==========================================
# 1. CİHAZ VE VERİ HAZIRLIĞI
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

df = pd.read_csv('time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Aykırı Değer (Outlier) Temizliği
df.loc[df['AT_load_actual_entsoe_transparency'] < 4000, 'AT_load_actual_entsoe_transparency'] = np.nan
df['AT_load_actual_entsoe_transparency'] = df['AT_load_actual_entsoe_transparency'].interpolate(method='time')

features = ['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual', 'AT_load_actual_entsoe_transparency']
data = df[features].dropna()

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.values)

# Günlük (24 Saatlik) Çoklu Tahmin İçin Kayan Pencere
def create_daily_sequences(data, lookback=48, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback), :]) # Geçmiş 48 saat (3 değişken)
        y_target = data[(i + lookback) : (i + lookback + horizon), :]
        y.append(y_target.flatten()) # Gelecek 24 saat (3 değişken düzleştirilmiş)
    return np.array(X), np.array(y)

lookback = 48  
horizon = 24   
X, y = create_daily_sequences(data_scaled, lookback, horizon)

# Veriyi %70 Eğitim, %15 Doğrulama, %15 Test Olarak Bölme
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).float()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).float()

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)

print("Veri Seti Hazırlandı!")

# ==========================================
# 2. MODEL MİMARİLERİ
# ==========================================
input_dim = 3
hidden_dim = 64 
layer_dim = 2
output_dim = 72 # 24 Saat * 3 Değişken

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

# ==========================================
# 3. ERKEN DURDURMALI EĞİTİM DÖNGÜSÜ
# ==========================================
def train_with_early_stopping(model, model_name, patience=5, epochs=30):
    print(f"\n--- {model_name} Modeli Eğitimi Başlıyor ---")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5) 
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Erken Durdurma! Model donduruldu (Epoch {epoch+1}).")
                break
                
    time_elapsed = time.time() - start_time
    print(f"Eğitim Tamamlandı! Süre: {time_elapsed:.0f}s | En İyi Val Loss: {best_val_loss:.5f}")
    
    model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).cpu().numpy()
        
    return preds, model

bilstm_model = BiLSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
gru_model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

bilstm_preds, best_bilstm = train_with_early_stopping(bilstm_model, "Bi-LSTM")
gru_preds, best_gru = train_with_early_stopping(gru_model, "GRU")

```

### Code Cell 2
```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score

# ==========================================
# 1. TAHMİNLERİ GERÇEK MW CİNSİNE ÇEVİRME
# ==========================================
# (Örnek Sayısı, 72) boyutundaki matrisi, (Örnek Sayısı * 24, 3) boyutuna getirip gerçek MW değerlerine döndürüyoruz.
def inverse_transform_daily(data_72d):
    data_reshaped = data_72d.reshape(-1, 3)
    data_mw = scaler.inverse_transform(data_reshaped)
    return data_mw

y_test_mw = inverse_transform_daily(y_test)
gru_preds_mw = np.clip(inverse_transform_daily(gru_preds), 0, None) # Üretim/Tüketim eksi olamaz

# ==========================================
# 2. 24 SAATLİK REGRESYON HATA METRİKLERİ
# ==========================================
def calculate_metrics(y_true, y_pred, feature_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    mean_actual = np.mean(y_true)
    wape = (mae / mean_actual) * 100 if mean_actual > 0 else 0
    
    print(f"[{feature_name}] -> RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | WAPE: %{wape:.2f}")

print("--- GRU MODELİ 24 SAATLİK GÜNLÜK TAHMİN HATA RAPORU ---")
calculate_metrics(y_test_mw[:, 0], gru_preds_mw[:, 0], "Güneş")
calculate_metrics(y_test_mw[:, 1], gru_preds_mw[:, 1], "Rüzgar")
calculate_metrics(y_test_mw[:, 2], gru_preds_mw[:, 2], "Tüketim")

# ==========================================
# 3. NET YÜK VE ŞEBEKE ESNEKLİĞİ (F1-SCORE) ANALİZİ
# ==========================================
# Net Yük = Tüketim (2. İndeks) - [Güneş (0. İndeks) + Rüzgar (1. İndeks)]
net_load_actual = y_test_mw[:, 2] - (y_test_mw[:, 0] + y_test_mw[:, 1])
net_load_gru = gru_preds_mw[:, 2] - (gru_preds_mw[:, 0] + gru_preds_mw[:, 1])

# KRİZ EŞİĞİ (Threshold): Net Yük 6000 MW'ı geçerse şebeke esnekliği tehlikede demektir (1), geçmezse güvendedir (0)
CRITICAL_THRESHOLD = 6000 

kriz_gercek = (net_load_actual > CRITICAL_THRESHOLD).astype(int)
kriz_tahmin = (net_load_gru > CRITICAL_THRESHOLD).astype(int)

f1 = f1_score(kriz_gercek, kriz_tahmin)
precision = precision_score(kriz_gercek, kriz_tahmin)
recall = recall_score(kriz_gercek, kriz_tahmin)

print("\n=====================================================")
print("     ŞEBEKE ESNEKLİK VE KRİZ TESPİT RAPORU (GRU)     ")
print("=====================================================")
print(f"Kritik Yük Eşiği      : {CRITICAL_THRESHOLD} MW")
print(f"Kriz Yakalama (Recall): %{recall*100:.2f} (Gerçek krizlerin yüzde kaçını 24 saat önceden bildi?)")
print(f"Hassasiyet (Precision): %{precision*100:.2f} (Kriz dediği anların yüzde kaçı gerçekten krizdi?)")
print(f"F1-Score (Başarı)     : %{f1*100:.2f} (Sınıflandırma modelinin genel başarı notu)")
print("=====================================================")
```

### Code Cell 3
```python
import matplotlib.pyplot as plt

# ==========================================
# GÖRSELLEŞTİRME: İLK 4 GÜN (96 SAAT)
# ==========================================
saat_sayisi = 96 
zaman_ekseni = range(saat_sayisi)

fig, axes = plt.subplots(4, 1, figsize=(16, 20))

# 1. NET YÜK GRAFİĞİ (Ana Odak Noktamız)
axes[0].plot(zaman_ekseni, net_load_actual[:saat_sayisi], label='Gerçek Net Yük', color='black', linewidth=2.5)
axes[0].plot(zaman_ekseni, net_load_gru[:saat_sayisi], label='GRU Tahmini Net Yük', color='red', linestyle='--', linewidth=2)
axes[0].axhline(y=CRITICAL_THRESHOLD, color='orange', linestyle='-.', linewidth=2, label=f'Kriz Eşiği ({CRITICAL_THRESHOLD} MW)')
axes[0].fill_between(zaman_ekseni, CRITICAL_THRESHOLD, net_load_actual[:saat_sayisi], where=(net_load_actual[:saat_sayisi] > CRITICAL_THRESHOLD), color='red', alpha=0.2, label='Gerçekleşen Kriz Anları')
axes[0].set_title('Şebeke Net Yükü ve Esneklik Krizi Tespiti (İlk 4 Gün)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Megawatt (MW)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle=':', alpha=0.7)

# 2. TÜKETİM GRAFİĞİ
axes[1].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 2], label='Gerçek Tüketim', color='black', linewidth=2)
axes[1].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 2], label='GRU Tahmini', color='blue', linestyle='--')
axes[1].set_title('Elektrik Tüketimi (Şebeke Yükü)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Megawatt (MW)', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':', alpha=0.7)

# 3. RÜZGAR GRAFİĞİ
axes[2].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 1], label='Gerçek Rüzgar', color='black', linewidth=2)
axes[2].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 1], label='GRU Tahmini', color='green', linestyle='--')
axes[2].set_title('Rüzgar Enerjisi Üretimi', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Megawatt (MW)', fontsize=12)
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle=':', alpha=0.7)

# 4. GÜNEŞ GRAFİĞİ
axes[3].plot(zaman_ekseni, y_test_mw[:saat_sayisi, 0], label='Gerçek Güneş', color='black', linewidth=2)
axes[3].plot(zaman_ekseni, gru_preds_mw[:saat_sayisi, 0], label='GRU Tahmini', color='orange', linestyle='--')
axes[3].set_title('Güneş Enerjisi Üretimi', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Zaman (Saat)', fontsize=12)
axes[3].set_ylabel('Megawatt (MW)', fontsize=12)
axes[3].legend(loc='upper right')
axes[3].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
```

### Code Cell 4
```python

```


================================================================================

## Notebook: Untitled1.ipynb
### Code Cell 1
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['GRU', 'LSTM', 'Bi-LSTM']
spatial_cv_wape = [12.22, 15.43, 11.58]       
smoothness_cv_wape = [12.36, 16.01, 12.43]    
forecast_wape = [20.18, 20.00, 22.11]         

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

rects1 = ax.bar(x - width, spatial_cv_wape, width, label='Mekansal CV (MSE)', color='#3b82f6')
rects2 = ax.bar(x, smoothness_cv_wape, width, label='Pürüzsüzlük Cezalı (Smoothness)', color='#10b981')
rects3 = ax.bar(x + width, forecast_wape, width, label='Gelecek Tahmini (Forecast)', color='#f59e0b')

ax.set_ylabel('WAPE Hatası (%) -- (Düşük olması iyi)', fontsize=12, fontweight='bold')
ax.set_title('Derin Öğrenme Modelleri Karşılaştırmalı Hata Analizi', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')

# Kural: Lejantın üst sütun yazılarıyla çakışmaması için grafiğin dışına (veya uygun boşluğa) konumlandırılması ve y-ekseni üst sınırının artırılması
ax.set_ylim(0, 25) 
ax.legend(loc='upper left', fontsize=10, frameon=True)

ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'%{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('model_karsilastirma_grafigi.png', dpi=300)
plt.show()
print("Düzeltilmiş model karşılaştırma grafiği 'model_karsilastirma_grafigi.png' olarak yeniden kaydedildi!")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: Untitled2.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veriyi Oku
df = pd.read_csv('forecast_sonuclari.csv')

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

hata_verileri = [df['GRU_Hata'], df['LSTM_Hata'], df['BiLSTM_Hata']]
etiketler = ['GRU Hataları', 'LSTM Hataları', 'Bi-LSTM Hataları']
renkler = ['#3b82f6', '#10b981', '#8b5cf6']

# Uyarıyı önlemek için 'labels' yerine 'tick_labels' kullanıyoruz
bplot = ax.boxplot(hata_verileri, patch_artist=True, tick_labels=etiketler, 
                   boxprops=dict(linewidth=1.5), 
                   medianprops=dict(color='black', linewidth=2),
                   flierprops=dict(marker='o', color='red', alpha=0.5))

for patch, color in zip(bplot['boxes'], renkler):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0, color='#ef4444', linestyle='--', linewidth=1.5, label='İdeal Sıfır Hata Çizgisi')

ax.set_title('Modellerin İstatistiksel Hata Kararlılığı ve Dağılım Analizi', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Tahmin Hatası (MW)', fontsize=11, fontweight='bold')
ax.set_xlabel('Derin Öğrenme Mimarileri', fontsize=11, fontweight='bold')

ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='#e5e7eb')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('hata_dagilimi_boxplot.png', dpi=300)
plt.show()
print("Hata dağılımı grafiği hatasız olarak yeniden oluşturuldu!")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: Untitled3.ipynb
### Code Cell 1
```python
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix

# ==========================================
# 1. VERİ YÜKLEME VE İZOLASYON (TEST SONUÇLARI)
# ==========================================
print("Operasyonel model tahmin sonuçları yükleniyor...")
df_sonuclar = pd.read_csv('forecast_sonuclari.csv')

# Doğru sütun isimleri ile verileri çekme
gercek_net_yuk = df_sonuclar['Gercek_Tuketim_MW'].values
gru_tahmin_yuku = df_sonuclar['GRU_Tahmin_MW'].values

# ==========================================
# 2. KRİZ EŞİĞİ VE BİNARY (İKİLİ) SINIFLANDIRMA DÖNÜŞÜMÜ
# ==========================================
KRIZ_ESIGI_MW = 6000

print(f"\nŞebeke Kriz Eşiği: {KRIZ_ESIGI_MW} MW olarak ayarlandı.")
print("Modellerin ürettiği sürekli (continuous) MW değerleri sınıflandırma matrisine dönüştürülüyor...")

# Vektörizasyon ile eşik değerin üstünü 1 (Kriz), altını 0 (Normal) olarak etiketliyoruz
y_true_kriz = (gercek_net_yuk > KRIZ_ESIGI_MW).astype(int)
y_pred_kriz = (gru_tahmin_yuku > KRIZ_ESIGI_MW).astype(int)

# ==========================================
# 3. İSTATİSTİKSEL KARAR DESTEK METRİKLERİ (RECALL & F1)
# ==========================================
recall_basarisi = recall_score(y_true_kriz, y_pred_kriz, zero_division=0) * 100
f1_basarisi = f1_score(y_true_kriz, y_pred_kriz, zero_division=0) * 100

# ==========================================
# 4. AKADEMİK RAPORLAMA VE ÇIKTI GÖSTERİMİ
# ==========================================
print("\n==================================================")
print("🔥 ÇOKLU MODEL KRİZ ÖNGÖRÜ BAŞARISI (BÖLGE B) 🔥")
print("==================================================")
print(f" -> Kritik Yük Yakalama (Recall) Başarısı : %{recall_basarisi:.2f}")
print(f" -> Kriz Öngörü Kararlılığı (F1-Score)    : %{f1_basarisi:.2f}")
print("==================================================\n")

print("--- Detaylı Sınıflandırma Raporu (Classification Report) ---")
rapor = classification_report(
    y_true_kriz, 
    y_pred_kriz, 
    target_names=['Normal Operasyon (0)', 'Şebeke Krizi (1)'],
    digits=4
)
print(rapor)

cm = confusion_matrix(y_true_kriz, y_pred_kriz)
print("--- Karmaşıklık Matrisi (Confusion Matrix) Operasyonel Özeti ---")
print(f"✅ Başarılı Tahmin (Normal Saatler)  : {cm[0][0]} saat")
print(f"⚠️ Yanlış Alarm (Kriz Yokken Var Dedi): {cm[0][1]} saat (False Positive)")
print(f"🚨 Atlanan Kriz (Kriz Varken Yok Dedi): {cm[1][0]} saat (Kritik Hata - False Negative)")
print(f"🎯 Doğru Yakalanan Kriz Anları       : {cm[1][1]} saat (True Positive)")
```

### Code Cell 2
```python

```


================================================================================

## Notebook: Untitled4.ipynb
### Code Cell 1
```python
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, confusion_matrix, precision_score

def operasyonel_kriz_raporu_olustur(y_gercek, y_tahmin, kriz_esigi=6000):
    """
    Şebeke yük tahminlerini (Regresyon) operasyonel kriz alarmlarına (Sınıflandırma) 
    dönüştürür ve modelin kriz yakalama performansını detaylı olarak hesaplar.
    """
    
    # 1. Veri Tiplerini Güvenceye Alma (PyTorch tensörleri veya Pandas serileri gelebilir)
    # Modelden çıkan verileri pürüzsüzce NumPy dizilerine çeviriyoruz
    gercek_np = np.array(y_gercek)
    tahmin_np = np.array(y_tahmin)
    
    # 2. Vektörel Eşikleme (Thresholding) Mantığı
    # 6000 MW ve üzeri durumlara 1 (Kriz), altındaki durumlara 0 (Normal) atanır
    kriz_gercek_sinif = (gercek_np >= kriz_esigi).astype(int)
    kriz_tahmin_sinif = (tahmin_np >= kriz_esigi).astype(int)
    
    # 3. İstatistiksel Metriklerin Hesaplanması
    recall = recall_score(kriz_gercek_sinif, kriz_tahmin_sinif)
    f1 = f1_score(kriz_gercek_sinif, kriz_tahmin_sinif)
    precision = precision_score(kriz_gercek_sinif, kriz_tahmin_sinif, zero_division=0)
    
    # 4. Karmaşıklık Matrisi (Confusion Matrix) Bileşenlerinin Ayrıştırılması
    # Jüriye asılsız alarmları ve kaçırılan krizleri net göstermek için
    tn, fp, fn, tp = confusion_matrix(kriz_gercek_sinif, kriz_tahmin_sinif).ravel()
    
    # 5. Profesyonel Konsol Çıktısı (Raporlama)
    print("=" * 60)
    print(f"⚡ OPERASYONEL KARAR DESTEK RAPORU (KRİZ EŞİĞİ: {kriz_esigi} MW) ⚡")
    print("=" * 60)
    print(f"Kriz Yakalama Başarısı (Recall) : %{recall * 100:.2f}")
    print(f"Kriz Öngörü Dengesi (F1-Score)  : %{f1 * 100:.2f}")
    print(f"Modelin Kesinliği (Precision)   : %{precision * 100:.2f}\n")
    
    print("-" * 60)
    print("KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ANALİZİ")
    print("-" * 60)
    print(f"[TP] Doğru Bilinen Krizler      : {tp} saat (Başarılı Alarm)")
    print(f"[FN] Kaçırılan Krizler          : {fn} saat (Gerçekte kriz olup alarm verilmeyen)")
    print(f"[FP] Asılsız Alarmlar           : {fp} saat (Normal saatte verilen yanlış kriz alarmı)")
    print(f"[TN] Doğru Bilinen Normaller    : {tn} saat (Sorunsuz işleyiş)")
    print("=" * 60)
    
    return recall, f1

# KULLANIM ÖRNEĞİ (Jüriye gösterilecek kısım):
# forecast_sonuclari.csv dosyasını Pandas ile okuyup fonksiyona gönderiyoruz
# df = pd.read_csv('forecast_sonuclari.csv')
# _ = operasyonel_kriz_raporu_olustur(df['Gercek_Tuketim'], df['GRU_Tahmini'], kriz_esigi=6000)
```

### Code Cell 2
```python
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, confusion_matrix, precision_score

# 1. Dosyayı oku
df = pd.read_csv('forecast_sonuclari.csv')

# 2. Sütunları eşleştir
gercek_degerler = df['Gercek_Tuketim_MW']
gru_tahminleri = df['GRU_Tahmin_MW']
kriz_esigi = 6000

# 3. Vektörel Eşikleme ve Dönüşüm
gercek_np = np.array(gercek_degerler)
tahmin_np = np.array(gru_tahminleri)

kriz_gercek_sinif = (gercek_np >= kriz_esigi).astype(int)
kriz_tahmin_sinif = (tahmin_np >= kriz_esigi).astype(int)

# 4. Metriklerin Hesaplanması
recall = recall_score(kriz_gercek_sinif, kriz_tahmin_sinif)
f1 = f1_score(kriz_gercek_sinif, kriz_tahmin_sinif)
precision = precision_score(kriz_gercek_sinif, kriz_tahmin_sinif, zero_division=0)
tn, fp, fn, tp = confusion_matrix(kriz_gercek_sinif, kriz_tahmin_sinif).ravel()

# 5. Raporlama
print("=" * 60)
print(f"⚡ OPERASYONEL KARAR DESTEK RAPORU (KRİZ EŞİĞİ: {kriz_esigi} MW) ⚡")
print("=" * 60)
print(f"Kriz Yakalama Başarısı (Recall) : %{recall * 100:.2f}")
print(f"Kriz Öngörü Dengesi (F1-Score)  : %{f1 * 100:.2f}")
print(f"Modelin Kesinliği (Precision)   : %{precision * 100:.2f}\n")
print(f"Doğru Bilinen Krizler (TP)      : {tp} saat")
print(f"Kaçırılan Krizler (FN)          : {fn} saat")
print(f"Asılsız Alarmlar (FP)           : {fp} saat")
print(f"Doğru Bilinen Normaller (TN)    : {tn} saat")
print("=" * 60)
```

### Code Cell 3
```python

```


================================================================================

