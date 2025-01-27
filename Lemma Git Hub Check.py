import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Make sure NLTK resources are downloaded (uncomment if necessary)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Load your data
last_rights_csv = pd.read_csv('C:/Users/stefa/OneDrive/Documents/Unstructured Data/Unstructured_Data_Acquisition_notes/last_statements.csv'
)

# Convert to string to ensure text handling
last_rights_csv = last_rights_csv.astype(str)

# Take a sample of the data
last_rights_sample = last_rights_csv.sample(1)
print(last_rights_sample)

# Ensure the correct column name (check if 'last_rights' is the right column)
# Print the column names to verify
print(last_rights_csv.columns)

# Initialize tokenizer and lemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function to lemmatize text
def lemmatize_text(statements):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(statements)]

last_rights_sample['last_right_lemma'] = last_rights_sample['statements'].apply(lemmatize_text)
    
#Print the lemmatized column
print(last_rights_sample['last_right_lemma'])
