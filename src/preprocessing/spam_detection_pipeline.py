import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import re
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Veriyi yükle
df = pd.read_csv('mail_data.csv')

# Sütun kontrolü
if 'Category' not in df.columns or 'Message' not in df.columns:
    raise ValueError("The dataset must contain 'Category' and 'Message' columns.")

# Eksik verileri doldurma
data = df.fillna('')

# Kategoriyi 0 ve 1 olarak etiketleme
data.loc[data['Category'] == 'spam', 'Category'] = 1
data.loc[data['Category'] == 'ham', 'Category'] = 0

# Özellik ve hedef ayrımı
X = data['Message']
Y = data['Category']

# Gürültü temizleme ve stemming fonksiyonu
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Özel karakterleri kaldır
    text = re.sub(r'\s+', ' ', text)  # Fazladan boşlukları kaldır
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Metni temizleme ve dönüştürme
X = X.apply(preprocess_text)

# Eğitim ve test verilerinin bölünmesi
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TfidfVectorizer ile özellik çıkarımı
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Hedef verilerini integer'a dönüştürme
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# PCA ile veri setini 2 boyuta indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_features.toarray())
X_test_pca = pca.transform(X_test_features.toarray())

# PCA sonucu için yeni KNN modelini eğit
knn_model_pca = KNeighborsClassifier(n_neighbors=5)
knn_model_pca.fit(X_train_pca, Y_train)

# Test verileriyle tahmin yap
Y_pred = knn_model_pca.predict(X_test_pca)

# Modelin doğruluğunu ölç
accuracy = accuracy_score(Y_test, Y_pred)
print("KNN Model Accuracy:", accuracy)

# Confusion Matrix'i hesapla
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Classification Report'u hesapla
report = classification_report(Y_test, Y_pred, target_names=['Ham', 'Spam'], output_dict=True)

# 1. Pencere: Confusion Matrix ve PCA Karar Sınırı Grafiği
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

# Confusion Matrix
titles = ['Ham', 'Spam']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=titles, yticklabels=titles, ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix')

# PCA Karar Sınırı
def plot_knn_decision_boundary(X_pca, Y, model, ax, title="KNN Decision Boundary with PCA"):
    h = 0.1  # Mesh grid adımı
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Mesh grid üzerindeki tüm noktalar için tahmin
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Karar sınırını çiz
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=30)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")


plot_knn_decision_boundary(X_train_pca, Y_train, knn_model_pca, axes[1], title="KNN Decision Boundary with PCA")

plt.tight_layout()
plt.show()

# 2. Pencere: Doğruluk Tablosu
report_data = [[k, v['precision'], v['recall'], v['f1-score'], v['support']] for k, v in report.items() if k in ['Ham', 'Spam']]
report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

table_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
for index, row in report_df.iterrows():
    table_data.append([row['Class'], round(row['Precision'], 2), round(row['Recall'], 2), round(row['F1-Score'], 2), int(row['Support'])])

table = ax.table(cellText=table_data, loc='center', cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
ax.set_title('Classification Report Table', fontsize=12)

plt.show()
