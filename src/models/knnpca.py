import pandas as pd
import re 
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from nltk.stem import PorterStemmer


df = pd.read_csv('mail_data.csv')

# etiket kontrolü
if 'Category' not in df.columns or 'Message' not in df.columns:
    raise ValueError("The dataset must contain 'Category' and 'Message' columns.")

# eksik verileri doldurma
data = df.fillna('')


data.loc[data['Category'] == 'spam', 'Category'] = 1
data.loc[data['Category'] == 'ham', 'Category'] = 0


X = data['Message']
Y = data['Category']

# gürültü ve stemming
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Ozel karakterleri kaldir
    text = re.sub(r'\s+', ' ', text)  # Fazladan bosluklari kaldir
    text = re.sub(r'\d+', '', text)  # Sayilari kaldir
    return ' '.join([stemmer.stem(word) for word in text.split()])

# fonksiyonu çalıştırma
X = X.apply(preprocess_text)

# Egitim ve test verilerinin bölünmesi
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF vektorizasyonu
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# PCA ile veri setini 2 boyuta indirgeme
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_features.toarray())
X_test_pca = pca.transform(X_test_features.toarray())

# Etiketlerin veri turunun donusturulmesi
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# K-Fold Cross Validation ile KNN modeli değerlendirme
kfold = KFold(n_splits=5, shuffle=True, random_state=3)
model = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(model, X_train_pca, Y_train, cv=kfold, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# KNN modelinin egitilmesi
model.fit(X_train_pca, Y_train)

# Test verilerinde tahmin yapma
prediction_on_test_data = model.predict(X_test_pca)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, prediction_on_test_data)
report = classification_report(Y_test, prediction_on_test_data, target_names=['Ham', 'Spam'], output_dict=True)

# Sonuclarin ekrana yazdirilmasi
print(f"KNN Model Accuracy: {accuracy_on_test_data * 100:.2f}%")

# 1. Grafik: Confusion Matrix ve PCA Karar Siniri
titles = ['Ham', 'Spam']
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

# Confusion Matrix
titles = ['Ham', 'Spam']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=titles, yticklabels=titles, ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix')

# PCA Karar Siniri
def plot_knn_decision_boundary(X_pca, Y, model, ax, title="KNN Decision Boundary with PCA"):
    h = 0.1  # Mesh grid adimi
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Mesh grid uzerindeki tum noktalar icin tahmin
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Karar sinirini ciz
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=30)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

plot_knn_decision_boundary(X_train_pca, Y_train, model, axes[1], title="KNN Decision Boundary with PCA")

plt.tight_layout()
plt.show()

# 2. Grafik: Dogruluk Tablosu
report_data = [[k, v['precision'], v['recall'], v['f1-score'], v['support']] for k, v in report.items() if k in ['Ham', 'Spam']]
report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

# Tablo icin veri hazirligi
table_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
for index, row in report_df.iterrows():
    table_data.append([row['Class'], round(row['Precision'], 2), round(row['Recall'], 2), round(row['F1-Score'], 2), int(row['Support'])])

# Tabloyu ekrana yazdirma
table = ax.table(cellText=table_data, loc='center', cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
ax.set_title('Classification Report Table', fontsize=12)

plt.show()
