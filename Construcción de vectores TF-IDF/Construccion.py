import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar los tweets preprocesados
ruta_datos = (r'C:\Users\emanu\Documents\Escuela\PROGRAMACION\python\Machine\Construcción de vectores TF-IDF\TWEETS_EMOCIONES_NEGATIVAS.txt')
with open(ruta_datos, 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Crear un objeto TfidfVectorizer para bigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
# Crear un objeto TfidfVectorizer para trigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 3))
# Crear un objeto TfidfVectorizer para cuatrigramas
#tfidf_vectorizer = TfidfVectorizer(ngram_range=(4, 4))
# Crear un objeto TfidfVectorizer para Unigramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Ajustar el vectorizador y transformar los datos de texto
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# Obtener el vocabulario (características)
feature_names = tfidf_vectorizer.get_feature_names_out()
# Convertir la matriz TF-IDF a DataFrame para mejor manejo
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Guardar los vectores TF-IDF en un archivo
df_tfidf.to_csv("vectores_tweets_Uni_emonegativas.txt", index=False)

print("Vectores TF-IDF guardados exitosamente.")

#Imprimir Vocabulario
print("Vacabulario (Caracteristicas): ")
print(feature_names)

#Matriz
print("\nMatriz TF-IDF")
print(tfidf_matrix)
