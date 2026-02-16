import pandas as pd
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('mail_data.csv')

if 'Category' not in df.columns or 'Message' not in df.columns:
    raise ValueError("The dataset must contain 'Category' and 'Message' columns.")


data = df.fillna('')


data.loc[data['Category'] == 'spam', 'Category'] = 1
data.loc[data['Category'] == 'ham', 'Category'] = 0


X = data['Message']
Y = data['Category']


stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\d+', '', text)  
    return ' '.join([stemmer.stem(word) for word in text.split()])


X = X.apply(preprocess_text)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


kfold = KFold(n_splits=5, shuffle=True, random_state=3)
model = LogisticRegression()

cv_scores = cross_val_score(model, X_train_features, Y_train, cv=kfold, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())


model.fit(X_train_features, Y_train)


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


print("Logistic Regression Model Accuracy on Test Data:", accuracy_on_test_data)


conf_matrix = confusion_matrix(Y_test, prediction_on_test_data)


titles = ['Ham', 'Spam']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=titles, yticklabels=titles)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


report = classification_report(Y_test, prediction_on_test_data, target_names=['Ham', 'Spam'], output_dict=True)


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
