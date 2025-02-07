import nltk
from nltk.corpus import gutenberg
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import streamlit as st

# Download necessary NLTK datasets
nltk.download('gutenberg')
nltk.download('punkt')

# 1. Data Collection
text = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# 2. Model Building
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=5, workers=4)

# 3. Visualization
vectors = model.wv.vectors
tsne = TSNE(n_components=2, random_state=0)
vectors_2d = tsne.fit_transform(vectors)
# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(model.wv.index_to_key[:100]):  # Show first 100 words
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
plt.show()

# 4. Streamlit App
st.title("Word Embeddings")
word = st.text_input("Enter a word:")
if word:
    try:
        embeddings = model.wv[word]
        st.write("Embeddings:", embeddings)
    except KeyError:
        st.write("Word not found in vocabulary.")
 
