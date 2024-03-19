# Import necessary libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pandas as pd
import re 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import string
from sklearn.tree import DecisionTreeClassifier
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")

# Read data
data = pd.read_csv('fake_job_postings.csv')

# Handle missing values
data = data.replace(np.nan, '', regex=True)

# Combine text columns
data['text'] = data[['title', 'department','company_profile','description','requirements','benefits']].apply(lambda x: ' '.join(x), axis = 1)

# Drop unnecessary columns
data.drop(['job_id', 'location','title','salary_range' ,'department','salary_range','company_profile','description','requirements','benefits'], axis=1, inplace=True)
data_columns = data.columns.tolist()

# Encode categorical variables
label_columns = ['telecommuting', 'has_company_logo', 'has_questions', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function']
lb_make = LabelEncoder()
for i in label_columns:
  data[i] = lb_make.fit_transform(data[i])

# Arrange columns
data_columns = data_columns[-1:] + data_columns[:-1]
data = data[data_columns]

# Define stopwords
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

# Function to remove URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))

# Function to remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))

# Function to remove HTML 
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',str(text))

# Function to remove punctuation
def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

# Function to decontract phrases
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Function for final preprocessing 
def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text

# Apply preprocessing steps
data['text'] = remove_URL(str(data['text']))
data['text'] = remove_emoji(str(data['text']))
data['text'] = remove_html(str(data['text']))
data['text'] = remove_punctuation(str(data['text']))
data['text'] = final_preprocess(str(data['text']))

df = data
# Split the data into training and testing sets
X = df['text']
y = df['fraudulent']

X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True,random_state=42,test_size=0.3,stratify=y)


# Define hyperparameters
max_words = 10000
max_len = 300
embedding_dim = 300
lstm_units = 200
dropout_rate = 0.1
EPOCHS = 8
BATCH_SIZE = 32

# Function to create an LSTM model
def LSTM_model(max_words, max_len, embedding_dim, lstm_units, dropout_rate):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create the model
LSTM_trained_model = LSTM_model(max_words, max_len, embedding_dim, lstm_units, dropout_rate)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequence, maxlen=max_len)
X_test_padded = pad_sequences(X_test_sequence, maxlen=max_len)

# Train the model
LSTM_trained_model.fit(X_train_padded, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
# Evaluate the model on the test set
loss, accuracy = LSTM_trained_model.evaluate(X_test_padded, y_test)
lstm_accuracy = accuracy


# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to convert a sentence to the vector using Universal Sentence Encoder
def convert_sen_to_vec_use(sentence):
    return embed([sentence])[0].numpy()

# Update the loop for converting data
converted_data_use = [convert_sen_to_vec_use(sentence) for sentence in data['text']]

# Update the dataframe creation
_1 = pd.DataFrame(converted_data_use)

scaler = StandardScaler()

data[['required_education', 'required_experience', 'employment_type']] = StandardScaler().fit_transform(data[['required_education', 'required_experience', 'employment_type']])

data.drop(["text"], axis=1, inplace=True)
main_data = pd.concat([_1,data], axis=1)

# Save the data for binary prediction
#data.to_csv('fake_Data.csv', index=False)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define models and hyperparameters
models = {
    'Random Forest': RandomForestClassifier(),
    'k-NN': KNeighborsClassifier(),
    'Histogram Gradient Boosting': HistGradientBoostingClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Define hyperparameter search space
param_grids = {
    "Random Forest": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None, 5, 10, 20],
    },
    "k-NN": {
        "clf__n_neighbors": [3, 5, 7, 10],
        "clf__weights": ["uniform", "distance"],
        "clf__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2, 0.3],
        "clf__max_depth": [3, 5, 7],
    },
    "Histogram Gradient Boosting": {
        "clf__loss": ["log_loss"],
        "clf__learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3],
        "clf__max_depth": [3, 5, 7, 9],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7, 9]
    },
    "Decision Tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__splitter": ["best", "random"],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": [None, "sqrt", "log2"]
    }
}

best_models = {}
results_dict = {}

# Iterate through models
for model_name, model in models.items():
    print(f"\n\n")
    print(f"Model: {model_name}")

    # Create the pipeline with the model
    pipeline = Pipeline([('clf', model)])

    # Perform hyperparameter tuning using RandomizedSearchCV
    param_grid = param_grids.get(model_name, {})
    search_cv = RandomizedSearchCV
    random_search = search_cv(
        pipeline,
        param_grid,
        n_iter=20,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    )

    # Perform cross-validation on the training and validation data to assess model performance
    cv_scores = cross_val_score(random_search, X_train, y_train, cv=5, scoring='accuracy')

    # Fit the model on the training and validation data
    random_search.fit(X_val, y_val)

    # Store the best model
    best_models[model_name] = random_search.best_estimator_

    # Store the results in the dictionary
    results_dict[model_name] = (random_search.best_params_, cv_scores)

results_list = []

# Populate the list with results
for model_name, (best_params, cv_scores) in results_dict.items():
    # Get the best model for this model_name
    best_model = best_models[model_name]

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get the test set F1-score
    f1_score_test = f1_score(y_test, y_pred, average='weighted')

    # Append results as a dictionary to the list
    results_list.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Test Set F1-Score': f1_score_test,
        'Accuracy': accuracy
    })

results_list.append({
    'Model': "LSTM",
    'Best Parameters': "NaN",
    'Test Set F1-Score': "NaN",
    'Accuracy': lstm_accuracy
})
# Create a DataFrame from the list of dictionaries
results_table = pd.DataFrame(results_list)

# Sort the DataFrame by Test Set F1-Score in descending order (best to worst)
results_table = results_table.sort_values(by='Accuracy', ascending=False)

# Print the results table
print(results_table)