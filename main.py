#İş Problemi
#Kural tabanlı müşteri segmentasyonu yöntemi RFM ile makine öğrenmesi yöntemi olan K-Means'in müşteri segmentasyonu için karşılaştırılması beklenmektedir.


#Veri Seti Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını
#içermektedir. Bu şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır. Şirketin müşterilerinin büyük çoğunluğu kurumsal müşterilerdir.


#GÖREV
#RFM metriklerine göre (skorlar değil) K-Means'i kullanarak müşteri segmentasyonu yapınız. Dilerseniz RFM metriklerinden başka metrikler de üretebilir ve bunları da kümeleme için kullanabilirsiniz

import pandas as pd
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
print(df_.head())
df = df_.copy()

###############################################################
# RFM
###############################################################

def create_rfm(dataframe):
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    return rfm


rfm = create_rfm(df)
print(rfm.head())

#Recency, Frequency ve Monetary kullanarak kümeleme yapmak istiyoruz. Bunun için değerleri standartlaştırmamız gerekiyor.
#K Means uzaklık temelli bir algoritmadır. Herhangi bir sütunda çok büyük değerler olduğu durumda segmenti belirleme anlamında problem yaratabilir.
#Büyük değer diğer değerleri domine edebilir. Ancak istenen şey recency, frequency ve monetary'nin eşit etkisinin olması.

#K means te amaç; kümelerin kendi içinde homojen bir şekilde, kendi dışında ise heterojen şekilde oluşması.
#Yani her bir küme birbirinden ne kadar uzaksa ve kendi içinde ne kadar benzerse amaca ulaşılmış olunur.

#Rastgele merkezler seçiliyor, bu merkezleri bizler giriyoruz.
#Kümeler kararlı hale gelene kadar merkezler sürekli değişiyor.
#Küme sayısını bilmemiz gerektiği için elbow yöntemini kullanıyoruz.

print("K-Means Clustering")
scaler = MinMaxScaler()
segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])
print(segment_data.head())

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(segment_data)
elbow.show()
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(segment_data) #elbow.elbow_value_--->6; kmeans'i 6 kümeye ayıracağız.   segment_data-->standartlaştırılmış frequency recency ve monetary değerleri
#k means modelini oluşturduktan sonra her bir gözlem için hangi gözlemin hangi kümeye ait olduğunu belirten labelları alt satırda elde ediyoruz.
segment_data["clusters"] = kmeans.labels_
print(f"Number of cluster selected: {elbow.elbow_value_}")
#6 değeri optimum. Kümeler arası farklılığı bilmek isteriz. Bu farklılığı yakalamak adına sadece 6 değeri değil 7 veya 8 kümeyi de deneyebiliriz. Ya da 6'dan daha
#düşük bir değeri de deneyebiliriz.

segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])

kmeans = KMeans(n_clusters=6).fit(segment_data)
segment_data["clusters"] = kmeans.labels_
print(segment_data["clusters"])

print(segment_data.head()) #hangi müşterinin hangi kümeye ait olduğunu tuttuğum clusters sütunu da eklendi

segmentation = rfm.merge(segment_data, on="Customer ID")
seg_desc = segmentation[["segment", "clusters", "recency", "frequency", "monetary"]].groupby(["clusters", "segment"]).agg(["mean", "count"])
print(seg_desc)
segmentation.to_csv("segmentation.csv")
