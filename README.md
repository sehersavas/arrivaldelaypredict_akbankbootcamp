#  ***Uçak Varış Gecikmeleri Sınıflandırması Tahmin Modellemesi | Supervised ML Projesi***

# PROJE HAKKINDA
Bu projede, ABD iç hat uçuşlarına ait 1936758 satır, 30 sütundan oluşan veri seti kullanılarak uçuşların varış gecikmeleri sınıflandırılmış ve tahmin modeli üretilmiştir. Amaç, çeşitli gecikme nedenlerini (hava durumu, kalkış gecikmesi, havayolu kaynaklı gecikme vb.) analiz ederek uçuşların zamanında, az gecikmeli, orta veya yüksek düzeyde gecikmeli olacağı önceden tahmin edilebilmesi hedeflenmektedir.
Bu öngörü, özellikle havayolu şirketlerinin operasyonel verimliliğini artırması, kaynak yönetimini optimize etmesi ve yolculara daha doğru bilgi verilmesi açısından kritik bir fayda sunmaktadır. Uygulama açısından bakıldığında, bu model sayesinde havayolları olası gecikmeleri önceden tahmin ederek uçak rotalamaları, personel planlaması, yer hizmetleri gibi birçok süreçte proaktif önlemler alabilir. Ayrıca, müşteri deneyimi açısından da gecikme riskine karşı ön bilgilendirme, alternatif uçuş önerileri gibi çözümler sunulabilir.

# PROBLEM TANIMI
Proje, uçuş varış gecikmelerinin tahmin edilmesine yönelik çok sınıflı bir sınıflandırma problemidir. Uçuşların zamanında gerçekleşip gerçekleşmeyeceği, havayolu sektöründe hem operasyonel planlama hem de müşteri memnuniyeti açısından kritik öneme sahiptir.

Hedef değişken olan "ArrDelay" verisi, uçuşların varış saatine göre dört farklı sınıfa ayrılarak etiketlenmiştir:

0: Zamanında veya erken varış (delay ≤ 0 dakika)
1: Az gecikmeli (1–50 dakika)
2: Orta düzeyde gecikmeli (51–90 dakika)
3: Yüksek düzeyde gecikmeli (> 90 dakika)

Model, hava durumu, kalkış gecikmesi, havayolu şirketi, uçuş mesafesi, gün/ay bilgisi gibi öznitelikleri kullanarak bir uçuşun bu sınıflardan hangisine ait olacağını tahmin etmeye çalışmaktadır. Böylece, potansiyel gecikmelerin önceden belirlenmesi sağlanarak daha etkin kaynak planlaması yapılması hedeflenmektedir.

# KULLANILAN YÖNTEMLER VE AKIŞ

*Modelde kullanılan yöntem*
Projede, çok sınıflı sınıflandırma problemini çözmek için Logistic Regression algoritması tercih edilmiştir. Logistic Regression çok sınıflı problemlerde de etkili sonuçlar verebilmektedir. Modelin hesaplama açısından hızlı, uygulaması kolay ve yorumlanabilirliği yüksek olması, büyük boyutlu veri setleriyle çalışırken önemli avantajlar sağlamaktadır.
Ayrıca veri setindeki değişkenlerin neredeyse hepsi sayısal olduğu için, Logistic Regression bu tür verilerle iyi çalışmaktadır. Aşırı karmaşık yapılar yerine, daha sade ve açıklanabilir bir model tercih edilerek, özellikle havayolu sektörü gibi kararların şeffaflıkla alınması gereken alanlarda güvenilirlik sağlanmıştır.


# *Verinin Yüklenmesi*
Veri, Kaggle üzerinden csv formantında indirilmiş, Pandas kullanılarak Jupyter Notebook'ta DataFrame’e aktarılmıştır:

``` import pandas as pd
df = pd.read_csv("/Users/sehersavas/Desktop/DelayedFlights.csv")
df.head()
```
# *Keşifsel Veri Analizi (EDA)*

Öncelikle verisetinin sayısal, istatiksel özelliklerine, eksik veri olup olmadığına, verilerin türlerine bakılmıştır.
```print(df.head())
print(f"\nVeri kümesinin boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print("\nSütunlar:")
print(df.columns.tolist())
print("\nVeri tipleri:")
print(df.dtypes)
print("\nEksik veri sayısı:")
print(df.isnull().sum())

df.describe(include='all' )
```
```console
Unnamed: 0  Year  Month  DayofMonth  DayOfWeek  DepTime  CRSDepTime  \
0           0  2008      1           3          4   2003.0        1955   
1           1  2008      1           3          4    754.0         735   
2           2  2008      1           3          4    628.0         620   
3           4  2008      1           3          4   1829.0        1755   
4           5  2008      1           3          4   1940.0        1915   

   ArrTime  CRSArrTime UniqueCarrier  ...  TaxiIn TaxiOut  Cancelled  \
0   2211.0        2225            WN  ...     4.0     8.0          0   
1   1002.0        1000            WN  ...     5.0    10.0          0   
2    804.0         750            WN  ...     3.0    17.0          0   
3   1959.0        1925            WN  ...     3.0    10.0          0   
4   2121.0        2110            WN  ...     4.0    10.0          0   

   CancellationCode  Diverted  CarrierDelay  WeatherDelay NASDelay  \
0                 N         0           NaN           NaN      NaN   
1                 N         0           NaN           NaN      NaN   
2                 N         0           NaN           NaN      NaN   
3                 N         0           2.0           0.0      0.0   
4                 N         0           NaN           NaN      NaN   

  SecurityDelay  LateAircraftDelay  
0           NaN                NaN  
1           NaN                NaN  
2           NaN                NaN  
3           0.0               32.0  
4           NaN                NaN  

[5 rows x 30 columns]

Veri kümesinin boyutu: 1936758 satır, 30 sütun

Sütunlar:
['Unnamed: 0', 'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

Veri tipleri:
Unnamed: 0             int64
Year                   int64
Month                  int64
DayofMonth             int64
DayOfWeek              int64
DepTime              float64
CRSDepTime             int64
ArrTime              float64
CRSArrTime             int64
UniqueCarrier         object
FlightNum              int64
TailNum               object
ActualElapsedTime    float64
CRSElapsedTime       float64
AirTime              float64
ArrDelay             float64
DepDelay             float64
Origin                object
Dest                  object
Distance               int64
TaxiIn               float64
TaxiOut              float64
Cancelled              int64
CancellationCode      object
Diverted               int64
CarrierDelay         float64
WeatherDelay         float64
NASDelay             float64
SecurityDelay        float64
LateAircraftDelay    float64
dtype: object

Eksik veri sayısı:
Unnamed: 0                0
Year                      0
Month                     0
DayofMonth                0
DayOfWeek                 0
DepTime                   0
CRSDepTime                0
ArrTime                7110
CRSArrTime                0
UniqueCarrier             0
FlightNum                 0
TailNum                   5
ActualElapsedTime      8387
CRSElapsedTime          198
AirTime                8387
ArrDelay               8387
DepDelay                  0
Origin                    0
Dest                      0
Distance                  0
TaxiIn                 7110
TaxiOut                 455
Cancelled                 0
CancellationCode          0
Diverted                  0
CarrierDelay         689270
WeatherDelay         689270
NASDelay             689270
SecurityDelay        689270
LateAircraftDelay    689270
dtype: int64
[2]:
Unnamed: 0	Year	Month	DayofMonth	DayOfWeek	DepTime	CRSDepTime	ArrTime	CRSArrTime	UniqueCarrier	...	TaxiIn	TaxiOut	Cancelled	CancellationCode	Diverted	CarrierDelay	WeatherDelay	NASDelay	SecurityDelay	LateAircraftDelay
count	1.936758e+06	1936758.0	1.936758e+06	1.936758e+06	1.936758e+06	1.936758e+06	1.936758e+06	1.929648e+06	1.936758e+06	1936758	...	1.929648e+06	1.936303e+06	1.936758e+06	1936758	1.936758e+06	1.247488e+06	1.247488e+06	1.247488e+06	1.247488e+06	1.247488e+06
unique	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	20	...	NaN	NaN	NaN	4	NaN	NaN	NaN	NaN	NaN	NaN
top	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	WN	...	NaN	NaN	NaN	N	NaN	NaN	NaN	NaN	NaN	NaN
freq	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	377602	...	NaN	NaN	NaN	1936125	NaN	NaN	NaN	NaN	NaN	NaN
mean	3.341651e+06	2008.0	6.111106e+00	1.575347e+01	3.984827e+00	1.518534e+03	1.467473e+03	1.610141e+03	1.634225e+03	NaN	...	6.812975e+00	1.823220e+01	3.268348e-04	NaN	4.003598e-03	1.917940e+01	3.703571e+00	1.502164e+01	9.013714e-02	2.529647e+01
std	2.066065e+06	0.0	3.482546e+00	8.776272e+00	1.995966e+00	4.504853e+02	4.247668e+02	5.481781e+02	4.646347e+02	NaN	...	5.273595e+00	1.433853e+01	1.807562e-02	NaN	6.314722e-02	4.354621e+01	2.149290e+01	3.383305e+01	2.022714e+00	4.205486e+01
min	0.000000e+00	2008.0	1.000000e+00	1.000000e+00	1.000000e+00	1.000000e+00	0.000000e+00	1.000000e+00	0.000000e+00	NaN	...	0.000000e+00	0.000000e+00	0.000000e+00	NaN	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00
25%	1.517452e+06	2008.0	3.000000e+00	8.000000e+00	2.000000e+00	1.203000e+03	1.135000e+03	1.316000e+03	1.325000e+03	NaN	...	4.000000e+00	1.000000e+01	0.000000e+00	NaN	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00
50%	3.242558e+06	2008.0	6.000000e+00	1.600000e+01	4.000000e+00	1.545000e+03	1.510000e+03	1.715000e+03	1.705000e+03	NaN	...	6.000000e+00	1.400000e+01	0.000000e+00	NaN	0.000000e+00	2.000000e+00	0.000000e+00	2.000000e+00	0.000000e+00	8.000000e+00
75%	4.972467e+06	2008.0	9.000000e+00	2.300000e+01	6.000000e+00	1.900000e+03	1.815000e+03	2.030000e+03	2.014000e+03	NaN	...	8.000000e+00	2.100000e+01	0.000000e+00	NaN	0.000000e+00	2.100000e+01	0.000000e+00	1.500000e+01	0.000000e+00	3.300000e+01
max	7.009727e+06	2008.0	1.200000e+01	3.100000e+01	7.000000e+00	2.400000e+03	2.359000e+03	2.400000e+03	2.400000e+03	NaN	...	2.400000e+02	4.220000e+02	1.000000e+00	NaN	1.000000e+00	2.436000e+03	1.352000e+03	1.357000e+03	3.920000e+02	1.316000e+03
11 rows × 30 columns
```

Hedef Değişkene ait Bilgilerin Edinilmesi
Hedef değişken olarak seçilecek olan varış gecikme dakikalarını gösteren "ArrDelay" sütunundaki verilere ilişkin bilgiler edinilmiştir. Hedef değişken ile özelliklerin korelasyonu değerlendirilmiş, en yüksek korelasyonu olan 15 özellik sıralanmıştır, bu özelliklerden bazıları ile hedef değişkenin ilişkileri grafikleştirilmiştir.

``` correlations = df_cleaned.corr(numeric_only=True)['ArrDelay'].sort_values(ascending=False)
print("ArrDelay ile korelasyonu yüksek değişkenler:\n", correlations.head(15))
```
```console
ArrDelay ile korelasyonu yüksek değişkenler:
 ArrDelay             1.000000
DepDelay             0.950323
CarrierDelay         0.501777
LateAircraftDelay    0.478127
NASDelay             0.387609
WeatherDelay         0.264604
TaxiOut              0.208441
TaxiIn               0.116314
DepTime              0.093979
ActualElapsedTime    0.083738
FlightNum            0.035803
CRSElapsedTime       0.031566
AirTime              0.027509
CRSArrTime           0.024998
CRSDepTime           0.017093
Name: ArrDelay, dtype: float64
```
"ArrDelay"'in kendisi ile olan korelasyonu dikkate alınmaz. Kalan değerler arasında DepDelay 0.950323 hedef değişken ile ciddi derecede doğrusal ilişkilidir, CarrierDelay,LateAircraftDelay,NASDelay hedef değişken ile orta derecede doğrusal ilişkilidir,WeatherDelay, TaxiOut, TaxiIn hedef değişken ile zayıf derecede de olsa doğrusal ilişkilidir. Kalan özellik sütunlarının korelasyon değerleri modelin anlamlı tahminde bulunmasına engel olabileceğinden ilerleyen basamaklarda veri temizliği aşamasında bu sütunlar verisetinden temizlenecektir.


# *Veri Temizliği*

*Eksik veriler temizlenmiş, veri temizliği sonrası kalan veriseti değerlendirilmesi yapılmıştır*

``` df_cleaned = df.dropna()
print("Yeni veri kümesinin boyutu:", df_cleaned.shape)
```

```console
Yeni veri kümesinin boyutu: (1247486, 30)
```

*Hedef değişken ile korelasyonu 0.01'den büyük olan 7 değer seçilmiş geri kalan özellik sütunları verisetinden çıkarılmış, kalan sütunlar ve eksik veri olup olmadığı kontrol edilmiştir.*

``` columns_to_keep = ['DepDelay', 'CarrierDelay', 'LateAircraftDelay', 
                   'NASDelay', 'WeatherDelay', 'TaxiOut', 'TaxiIn', 'ArrDelay']

df_cleaned = df.dropna()

df_cleaned = df_cleaned[columns_to_keep]

print("Kalan sütunlar:")
print(df_cleaned.columns.tolist())

print("Yeni df_cleaned boyutu:", df_cleaned.shape)
print("Eksik veri var mı?:", df_cleaned.isnull().sum().sum())
```
```console
Kalan sütunlar:
['DepDelay', 'CarrierDelay', 'LateAircraftDelay', 'NASDelay', 'WeatherDelay', 'TaxiOut', 'TaxiIn', 'ArrDelay']
Yeni df_cleaned boyutu: (1247486, 8)
Eksik veri var mı?: 0
```

*Daha önce keşifsel veri analizinde keşfedilen uç değerler temizlendi.*

```threshold = 200
df_filtered = df_cleaned[df_cleaned['ArrDelay'] <= threshold]

print("Yeni df_filtered boyutu:", df_filtered.shape)
print("Eksik veri var mı?:", df_filtered.isnull().sum().sum())
```
```console
Yeni df_filtered boyutu: (1204914, 8)
Eksik veri var mı?: 0
```

# *Veri Ön İşleme*

Hedef değişken olarak belirlenen "ArrDelay" sütunundan kategoriler oluşturulmuştur. Modelde logistic regression kullanılması planlandığı için hedef değişken if, elif ve else fonksiyonları kullanılarak 4 sınıfa bölünmüştür.

ArrDelay değeri;
0 veya 0'dakikadan küçükse veri 0 sınıfına,
0 dakikadan büyük 50 dakikadan küçük veya 50 dakikaysa 1 sınıfına,
50 dakikadan büyük 90 dakikadan küçük veya 90 dakika ise 2 sınıfına,
90 dakikadan büyük ise 3 sınıfına,
ait olacaktır.

"ArrDelayClass" adında yeni bir sınıf oluşturulmuştur.

``` def delay_class_4(x):
    if x <= 0:
        return 0
    elif x <= 50:
        return 1
    elif x <= 90:
        return 2
    else:
        return 3

df_filtered = df_filtered.copy()
df_filtered['ArrDelayClass'] = df_filtered['ArrDelay'].apply(delay_class_4)
```
*Daha önceden temizlenen değişkenler karışıklık olmaması için yeniden tanımlanmış, ardından x bağımsız değişkenleri DepDelay', 'CarrierDelay', 'LateAircraftDelay', 'NASDelay', 'WeatherDelay', 'TaxiOut', 'TaxiIn' olarak, y hedef değişkeni ise "ArrDelayClass" olarak tanımlanmıştır. Verilerin değerlerinin türleri farklı olabileceğinden 0,1 aralığında normalize edilmiştir.*

``` from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

features = ['DepDelay', 'CarrierDelay', 'LateAircraftDelay', 
            'NASDelay', 'WeatherDelay', 'TaxiOut', 'TaxiIn']

x = df_filtered[features]
y = df_filtered['ArrDelayClass']

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_scaled_df = pd.DataFrame(x_scaled, columns=features)
```

*Makine öğrenmesi kullanılarak yapılacak regresyon modelinin eğitilmesi için verisetindeki hem bağımsız hem de hedef değişken verilerin %80'i eğitim verisi, modelin başarısının test edilmesi için verisetindeki verilerin %20'si test verisi olarak bölünmüştür.*

``` from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x_scaled_df, y, test_size=0.2, random_state=42, stratify=y )
```


# *Model Oluşturulması, Uygulanması, Test Edilmesi, Optimizasyonu*

*Logistic Regression kullanılarak model oluşturulmuş ve tahminleme uygulanmıştır.*
``` from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

*Modelin Test Edilmesi*

``` from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipeline = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000, random_state=42))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring='accuracy', n_jobs=-1)

print("Cross Validation Doğruluk Skorları:", cv_scores)
print("Ortalama Doğruluk:", cv_scores.mean())
print("Standart Sapma:", cv_scores.std())
```
```console
Cross Validation Doğruluk Skorları: [0.98967147 0.99043916 0.99010304 0.98946399 0.98970048]
Ortalama Doğruluk: 0.9898756258329222
Standart Sapma: 0.0003495967047013123
```
Cross validation sonuçlarına göre modelin doğruluk oranları beş farklı katmanda oldukça tutarlı ve yüksek seyrettiği, doğruluk skorlarının %98.96 ile %99.04 arasında değişmekte olduğu, ortalama doğruluk %98.99 seviyesinde olduğu, standart sapmanın çok düşük olması (0.00035 civarı) farklı katmanlardaki performansın birbirine çok yakın olduğu, yani modelin kararlı ve genelleme kapasitesinin yüksek olduğu gözlenmektedir. Bu verilerle aşırı öğrenme (overfitting) riskinin düşük olduğu söylenebilir.

*Modelin Hiperparametre Optimizasyonu Yapılarak Yeniden Uygulanması*
``` from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__max_iter': [200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(x, y)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk:", grid_search.best_score_)
```
```console
En iyi parametreler: {'logisticregression__C': 10, 'logisticregression__max_iter': 200}
En iyi doğruluk: 0.9965258930591936
```

Modelin en iyi doğruluk değerini C=10 ve max_iter=200 parametreleriyle elde ettiği görüldü. Bu doğruluk oranı %99.65 gibi çok yüksek bir değeri göstermekte olup modelin verideki sınıfları neredeyse doğru şekilde tahmin ettiği tespit edilmiştir.

C=10 değeri, modelin aşırı uyuma (overfitting'i önlemeye) daha az ağırlık verip, veriye daha esnek uyum sağladığını ifade etmektedir. max_iter=200 ise modelin eğitim sürecinde yeterli sayıda iterasyon yaparak parametreleri optimize ettiğini göstermektedir.

Bu yüksek doğruluk seviyesi, modelin sınıflandırma başarısının çok iyi olduğunu gösterse de, aşırı uyum (overfitting) riskinin olup olmadığını kontrol etmek gerekmektedir. Bu nedenle diğer performans metrikleri (precision, recall, f1-score) da incelenecektir.

# *En iyi performans gösteren değerlerle uygulanan modelin değerlendirilmesi*
``` from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()

# Performans metrikleri
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("Kesinlik (Precision) (macro):", precision_score(y_test, y_pred, average='macro'))
print("Duyarlılık (Recall) (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Skoru (macro):", f1_score(y_test, y_pred, average='macro'))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
```




![image](https://github.com/user-attachments/assets/33ddef0b-0663-4a8d-be7c-2457a17c5308)

```console
Doğruluk (Accuracy): 0.9960080171630364
Kesinlik (Precision) (macro): 0.995652481470341
Duyarlılık (Recall) (macro): 0.9952990802595668
F1 Skoru (macro): 0.9954756334827012

Classification Report:

              precision    recall  f1-score   support

           1     0.9970    0.9975    0.9973    142406
           2     0.9919    0.9915    0.9917     56417
           3     0.9980    0.9968    0.9974     42160

    accuracy                         0.9960    240983
   macro avg     0.9957    0.9953    0.9955    240983
weighted avg     0.9960    0.9960    0.9960    240983
```
Hiperparametre optimizasyonu sonucu elde edilen en iyi modeli (best_model) seçildi, test veri setindeki özelliklerden (x_test) tahminler yapıldı ve bu tahminlerle gerçek test hedef değerlerini (y_test) karşılaştırıldı. Tahminlerin doğruluğunu anlamak için karışıklık matrisi (confusion_matrix) oluşturuldu ve bu matris görsel olarak seaborn kütüphanesiyle ısı haritası şeklinde gösterildi.

Daha önce hedef değişken (0,1,2,3) olmak üzere 4 sınıfa ayrılmıştı ancak 0 sınıfına ait veri olmadığından karışıklık ısı haritasında temsil edilmediği düşünülmektedir.

Karışıklık haritasına bakıldığında ; sınıf 1 için 350 örneğin 2 olarak hatalı tahmin edildiği, 3 olarak hatalı tahmin olmadığı, sınıf 2 için 395 örneğin 1 olarak hatalı , 84 örneğin 3 olarak hatalı tahmin edildiği, sınıf 3 için 27 örneğin 1 olarak hatalı , 106 örneğin 2 olarak hatalı tahmin edildiği, gözlenmiştir.

Karışıklık ısı haritası çıktısına bakılarak en başarılı performansın sınıf 1 tahminlemesinde olabileceği, en az başarılı performansın sınıf 2 tahminlemesinde olabileceği gözlenmekte olup modelin performansına ilişkin kesin yorum yapılabilmesi için diğer metriklerin de değerlendirilmesi gerekmektedir.

Ek olarak model performansını ölçmek için doğruluk (accuracy), kesinlik (precision), duyarlılık (recall) ve F1 skoru gibi temel sınıflandırma metrikleri hesaplanmış olup her sınıf için ayrıntılı değerlendirme raporu (classification_report) yazdırıldı. Bu sonuçlara bakıldığında; Sınıf 1 için precision (kesinlik) ve recall (duyarlılık) %99.7 civarında, Sınıf 2 için precision ve recall biraz daha düşük (%99.1), Sınıf 3 için precision ve recall yine %99.8 seviyesinde, gözlenmiş olup en yüksek performansı sınıf 3'ün verdiği görülmektedir.

Macro average değerleri, her sınıfın eşit ağırlıklı ortalamasını temsil etmekte olup yaklaşık %99.5 seviyesindedir.

Weighted average’ın accuracy ile neredeyse aynı çıkması, modelin sınıf dağılımını başarıyla öğrenebildiğini göstermekte olup dengesizliğin performansa olumsuz etki etmediği söylenebilir.

Sonuç olarak, bağımsız değişkenlere bağlı olan uçak varış gecikmelerinin kategorik olarak tahminine ilişkin modelin başarılı çalıştığı gözlenmiştir.

# *SONUÇ VE GELECEK ÇALIŞMALAR*
Projede, uçuş varış gecikmelerini sınıflandırmak üzere Logistic Regression algoritması ile temel bir sınıflandırma modeli geliştirilmiş ve model metriklere göre yüksek performans sergilemiştir. Ancak, gerçek dünya uygulamalarında daha yüksek başarı, genelleme ve dayanıklılık için projenin bazı yönleri geliştirilmeye açıktır.

Projede yalnızca var olan sütunlar kullanılmıştır. Ancak modele harici bilgiler de entegre edilebilir, örneğin yoğunluk verileri (havaalanı bazında saatlik uçuş yoğunluğu), mevsimsel ve tatil bilgileri (bayramlar, resmi tatiller, sezonluk yoğunluk) benzeri. Bu tür ek bilgiler modele bağlamsal anlam katar ve tahmin doğruluğunu artırabilir.

Model şu an ilkel düzeyde olup bir API üzerinden web tabanlı bir arayüzle entegre edilebilir, uygulama oluşturulabilir. Böylece havayolu şirketi yahut hizmet alan yolcuların uçuş bilgilerini girerek sistemden gecikme tahmini alabilmesi sağlanabilir.

Model Kaggle url : https://www.kaggle.com/code/sehersava/flightarrivaldelaymodel-logistic-regression-ipynb
