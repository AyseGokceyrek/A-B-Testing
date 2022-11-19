#############################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#############################################

#############################################
# İş Problemi / Business Problem
#############################################
# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü
# olan "averagebidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi
# ve average bidding'in maximum bidding'den daha fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak
# istiyor. A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi
# bekliyor. Bombabomba.com için nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase
#  metriğine odaklanılmalıdır.

#############################################
# Veri Seti Hikayesi
#############################################
# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları
# gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı
# veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır.
# Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# 4 Değişken 40 Gözlem 26 KB
# Impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç

#############################################
# PROJE GÖREVLERİ / PROJECT TASKS
#############################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#############################################
# GÖREV 1: Veriyi Anlama ve Hazırlama / Understanding and Preparing Data
#############################################
# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df_control = pd.read_excel("C:/Users/Lenovo/PycharmProjects/datasets/ab_testing.xlsx", sheet_name="Control Group")
df_control.head()
df_test = pd.read_excel("C:/Users/Lenovo/PycharmProjects/datasets/ab_testing.xlsx", sheet_name="Test Group")
df_test.head()
# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_test.describe().T
df_control.describe().T

df_control.info()
df_test.info()

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df_test["Group"] = "Average"
df_control["Group"] = "Maxi"
# İki verisetimizi birleştirdiğimizde hangi verinin nereden geldiğinin görmek adına 'Bidding' sütunu oluşturulmuştur.

df = pd.concat([df_control, df_test])
df.head(15)
df.tail(15)
df.describe().T
#############################################
# GÖREV 2: A/B Testinin Hipotezinin Tanımlanması
#############################################
# Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2 (Kontrol ve test grupları için kazanç ortalamalarındaki farkın istatistiksel olarak anlamı yoktur)
# H1 : M1!= M2 (.... vardır)

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df[df["Group"] == "Average"]["Purchase"].mean()
df[df["Group"] == "Maxi"]["Purchase"].mean()


"""
 Control ve test gruplarından gelen purchase değerlerinin dağılımını grafikte gözlemlediğimiz gibi birbirine yakınlık
göstermektedir. Bu yakınlığın tesadüfen mi meydana gelmekte yoksa istatistiksel olarak bir anlam ifade mi ediyor 
kontrol edelim!
"""

data_vis = (df.groupby("Group").agg({"Purchase": "mean"}))
data_vis = data_vis.reset_index()

# bar plot
fig, ax = plt.subplots(figsize=(18, 10))
sns.barplot(x='Group', y='Purchase', data=data_vis, ax=ax)
plt.show()

data_vis = (df.groupby("Group").agg({"Purchase": "mean"})).plot(kind='bar')
plt.show()

# boxplot
sns.boxplot(x=df["Group"], y=df["Purchase"], hue=df["Group"], data=df)
plt.show()

# 2. deneme
sns.set_style('whitegrid')
ax = sns.boxplot(x='Group', y='Purchase', data=data_vis)
ax = sns.stripplot(x="Group", y="Purchase", data=data_vis)
plt.show()

# 3. deneme
sns.boxplot(x='Group', y='Purchase', data=data_vis,
            linewidth=2, showmeans=True,
            meanprops={"marker": "*","markerfacecolor": "green", "markeredgecolor": "pink"})
#############################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#############################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ?
# Elde edilen p-value değerlerini yorumlayınız.

# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-valuedeğerlerini yorumlayınız.

test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Average", "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Maxi", "Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

"""
Kontrol(Maxi) ve Test(Average) gruplarının Purchase değişkeninin p-value değerlerine baktığımız da 0.05'ten büyük olduğunu görüyoruz. Bu da 
Normallik varsayımı H0 hipotezimizin reddedilemeyeceğinin göstergesidir. Yani Normal Dağılım varsayımı sağlanmaktadır.
"""

test_stat, pvalue = levene(df_test["Purchase"].dropna(), df_control["Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

"""
Kontrol ve Test gruplarının purchase değişkenine göre pvalue değerlerine baktığımızda burada da 0.05'ten büyük olduğu için 
Varyans homojenliği testimizde varyansların homojen olduğunu gözlemliyoruz. Dolayısı ile buradan ttesti uygulayacağımız
sonucunu elde etmiş oluyoruz.

Burada ilk başta oluşturulan kontrol ve test dataframe'lerinden purchase değerleri çağrılmıştır. Normallik varsayımındaki gibi 
en son birleştirilmiş df dataframe'den de çağrılabileceği gösterilemek amaçlı iki farklı şekilde yapılmıştır. 
"""

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

test_stat, pvalue = ttest_ind(df.loc[df["Bidding"] == "Average", "Purchase"],
                              df.loc[df["Bidding"] == "Maxi", "Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.
"""
Yapılan ttesti sonucu almış olduğumuz pvalue= 0.3493 > 0.05 olduğu için başta oluşturmuş olduğumuz H0 reddedilmez.
Yani Kontrol ve test grupları için kazanç ortalamalarındaki farkın istatistiksel olarak anlamı yoktur. 
"""
#############################################
# GÖREV 4: Sonuçların Analizi
#############################################
# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
"""
Hipotezimizi oluşturduktan sonra, Normallik Varsayımının ve Varyans Hojenliğinin kontrolü yapılmış ve 
H0'ları reddedilememiştir. Varsayımlarımız sağlandığı için bağımsız iki örneklem T-Testi(Parametrik test) uygulanmıştır.
"""
# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz
"""
bombabomba.com'un facebook sayfasındaki alternatif teklif verme ile hali hazırda kullandıkları teklif verme türlerinin,
satın alınan ürün sayısını bariz bir şekilde etkilemediğini gözlemliyoruz. Belki 1 ay daha veri toplamak gerekebilir.
Sonrasında, değişiklikler tekrardan gözlemlenebilir. 
"""

