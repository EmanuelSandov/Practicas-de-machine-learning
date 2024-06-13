import stanza
import re

stanza.download('es')  # Descargar el modelo en español

nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
#Eliminacion de menciones
patron_1= re.compile(r'@([A-Za-z][A-Za-z0-9_]*)?')
#Eliminacion de URL
patron_2= re.compile(r'(http||https)://[A-Za-z0-9-_]*.[A-Za-z]*(/([A-Za-z0-9]*)?)?')
#Patron para eliminar caracteres speciales
patron_3= re.compile(r'[:\-\/\(\)\_\;\.\!\?\[\]\¿\¡\,\'\"\$\#]')
#Patron para palabras vacias
patron_4 = re.compile(r'\b(y|e|o|u|con|de|en|por|para|a|el|la|los|las|un|una|unos|unas|al|del)\b')
#Diccionario para emogis
# Diccionario para convertir emojis en palabras
emojis_a_palabras = {
    ':-/': 'inseguro',
    't.t': 'llorando',
    ':)': 'feliz',
    ':(': 'triste',
    '-.-': 'disconforme',
    ':b': 'sacar_lengua',
    ':o': 'sorprendido',
    ':f': 'ceño',
    ':s': 'inseguro',
    ':p': 'sacar_lengua'
}

# Ruta al archivo de texto
archivo = r"C:\Users\emanu\Documents\Escuela\PROGRAMACION\python\Machine\Construcción de vectores TF-IDF\tweets_emo_negativas.txt"
ar_salida="resultado.txt"

# Leer el contenido del archivo
with open(archivo, 'r', encoding='utf-8') as file:
    texto = file.read().lower()

# Reemplazar emojis por palabras
for emoji, palabra in emojis_a_palabras.items():
    texto = re.sub(re.escape(emoji), palabra, texto)

# Procesar el texto con Stanza
doc = nlp(texto)

# Abrir el archivo resultados.txt en modo escritura
with open("TWEETS_EMOCIONES_NEGATIVAS.txt", "w", encoding="utf-8") as archivo_resultados:
    # Definimos una variable de número de oración
    noracion = 0
    for sent in doc.sentences:
        noracion += 1
        archivo_resultados.write(f'===== Frase {noracion} tokens =====\n')
        for word in sent.words:
            #Guardamos la linea como se solicita en el documento id - palabra - lema
             #Para quitar las menciones
            if patron_1.match(word.text) :
                continue
            if patron_2.match(word.text):
                continue
            if patron_3.match(word.text):
                continue
            if patron_4.match(word.text):
                continue
            archivo_resultados.write(f'id: {word.id}'.ljust(8) + f'Palabra: {word.text}'.ljust(25) + f'Lema: {word.lemma}\n')

