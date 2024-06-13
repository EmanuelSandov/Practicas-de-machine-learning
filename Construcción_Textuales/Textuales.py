from sklearn.feature_extraction.text import TfidfVectorizer

# Ruta del archivo
ruta_datos = (r"C:\Users\emanu\Documents\Escuela\PROGRAMACION\python\Machine\Construcción_Textuales"
              r"\tweets_emo_negativas.txt")

# Leer los tweets del archivo, cada línea es un tweet
with open(ruta_datos, 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Crear un objeto TfidfVectorizer para unigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
# Crear un objeto TfidfVectorizer para bigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
# Crear un objeto TfidfVectorizer para trigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 3))
# Crear un objeto TfidfVectorizer para cuatrigramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(4, 4))

# Ajustar el vectorizador con los datos de los tweets y transformarlos
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# Obtener el vocabulario (características)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convertir la matriz TF-IDF a una matriz densa
dense_matrix = tfidf_matrix.todense()

# Preparar para guardar los resultados en un archivo de texto
#with open("unigrama_tweets_prepro.txt", "w", encoding="utf-8") as output_file:
#with open("bigrama_tweets_prepro.txt", "w", encoding="utf-8") as output_file:
#with open("trigrama_tweets_prepro.txt", "w", encoding="utf-8") as output_file:
with open("cuatrigrama_tweets_prepro.txt", "w", encoding="utf-8") as output_file:
    for row in dense_matrix:
        # Extraer los índices de los unigramas con valores TF-IDF mayores que cero
        nonzero_indices = row.nonzero()[1]
        # Extraer los unigramas correspondientes
        unigramas = [feature_names[index] for index in nonzero_indices if row[0, index] > 0]
        # Unir los unigramas con un espacio y escribir en el archivo
        output_file.write(" ".join(unigramas) + "\n")
