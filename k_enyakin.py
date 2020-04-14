import os
import math
import pandas as pd                                                                                                         # Gerekli kütüphaneleri import ettik
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(" ")
print(" ")

try:
    uzaklik = int(input("\t\t...UZAKLIK ÖLÇÜTÜNÜ BELİRLEYİNİZ...\n"
                        "\t\t\tMİNKOWSKİ için\t\t0 \n"
                        "\t\t\tMANHATTAN için\t\t1 \n"
                        "\t\t\tÖKLİD için\t\t2 \n"
                        "\t\t\tCHEBYSCHEV için\t\t3 \n"
                        "\t\t....GİRİNİZ....\n"
                        ">>> "))
except:
    print("LÜTFEN 0 - 1 - 2 - 3 SAYILARINDAN BİRİNİ GİRİNİZ...")
    uzaklik = int(input("\t\t...UZAKLIK ÖLÇÜTÜNÜ BELİRLEYİNİZ...\n"
                        "\t\t\tMİNKOWSKİ için \t\t0 \n"
                        "\t\t\tMANHATTAN için \t\t1 \n"
                        "\t\t\tÖKLİD için \t\t2 \n"
                        "\t\t\tCHEBYSCHEV için \t\t3 \n"
                        "\t\t....GİRİNİZ...\n"
                        ">>> "))

if uzaklik == 0:
    print("MİNKOWSKİ UZAKLIK ÖLÇÜTÜ SEÇİLDİ...")
if uzaklik == 1:
    print("MANHATTAN UZAKLIK ÖLÇÜTÜ SEÇİLDİ...")
if uzaklik == 2:
    print("ÖKLİD UZAKLIK ÖLÇÜTÜ SEÇİLDİ...")
if uzaklik == 3:
    print("CHEBYSCHEV UZAKLIK ÖLÇÜTÜ SEÇİLDİ...")

yol = input("Analiz için dosya ismini giriniz : ")
PATH = str("./" + yol)
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print("Dosyamız mevcut ve okunabilir durumdadır.")
else:
    print("Dosyamız mevcut değil ya da okunabilir durumda değildir.")
    yol = input("Analiz için dosya ismini giriniz : ")

try:
    kdeger = int(input("K DEĞERİNİ GİRİNİZ : "))
except:
    print("LÜTFEN SAYI GİRİNİZ...")
    kdeger = int(input("K DEĞERİNİ GİRİNİZ : "))
else:
    print("**********************************************")

data = pd.read_csv(yol)                                                                                                     # csv dosyamızı okuduk.
bol = data.iloc[:, -1:].values                                                                                              # Bağımlı Değişkeni ( bol) bir değişkene atadık
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:-1], bol, test_size=0.33,random_state=0)                 # veri kümemizi parçalara ayırdık
knn = KNeighborsClassifier(n_neighbors=kdeger, metric='minkowski',p=uzaklik)                                                # k değeri = n_neighbors değeridir. default 5 tir.
                                                                                                                            # k ve p değerlerini kullanıcıdan aldık... p == 2 ise öklid 
knn.fit(x_train, y_train.ravel())

tahmin = knn.predict(x_test)                                                                                                # test verimizi tahmin et dedik ve tahmin değişkenine atadık
matrisimiz = confusion_matrix(y_test,tahmin)                                                                                # tahminimizin karmaşık matrisini oluşturduk ve matrisimiz adlı değişkene atadık
print("\t\tMATRİSİNİZ : \n"
      "",matrisimiz)
dogruluk = accuracy_score(y_test,tahmin)                                                                                    # tahmin değişkenimizin doğruluk sonucunu getir ve dogruluk adlı değişkene ata dedik
yuzde = dogruluk * 100
print("YUZDE : %", yuzde, " DOĞRULUK ORANI BULUNMAKTA ")
toplam = matrisimiz.sum()
print("TOPLAM ",toplam," VERİ İLE İŞLEM YAPILMIŞTIR")
print("YANLIŞ YAPILAN TAHMİN ADETİ ",matrisimiz[1,2],"TANEDİR")
son = input("\n\t\tKAPATMAK İÇİN HERHANGİ BİR TUŞA BASINIZ...")


