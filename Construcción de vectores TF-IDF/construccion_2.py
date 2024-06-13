from sklearn.feature_extraction.text import TfidfVectorizer

# Datos de ejemplo (lista de documentos)
corpus = ['Este es un ejemplo de texto',
          'Otro ejemplo de texto',
          'Algunos ejemplos son buenos' ]
# Crear un objeto TfidfVectorizer
tfidf_vectorizer   =   TfidfVectorizer(ngram_range=(1,2))  #considera   unigramas   y bigramas.
# Ajustar el vectorizador y transformar los datos de texto
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
# Obtener el vocabulario (características)
feature_names = tfidf_vectorizer.get_feature_names_out()
# Imprimir el vocabulario
print("Vocabulario (Características):")
print(feature_names)
# Imprimir la matriz TF-IDF
print("\nMatriz TF-IDF:")
print(tfidf_matrix.toarray())