<h1>Word2Vec</h1>

<h5>Dans ce devoir, nous allons entraîner Word2vec avec la base de données movies_metadata présente sur https://www.kaggle.com/rounakbanik/the-movies-dataset. </h5>


<h5>Word2Vec est un groupe de modèles utilisé pour le plongement lexical. 
Le plongement lexical est une méthode d'apprentissage d'une représentation de mots sous forme de vecteurs, utilisée notamment en traitement automatique des langues. Cette technique permet de représenter chaque mot d'un dictionnaire par un vecteur de nombres réels, autrement dit une liste de nombres. </h5>

Nous importons les bases de données pour les transformer en dataframe grâce à la librairie pandas


```python
import pandas as pd

movies_metadata=pd.read_csv(r"C:\Users\lenovo\OneDrive\Documents\M2\FOND BIG DATA\movies_dataset\movies_metadata.csv")

```

    C:\Users\lenovo\AppData\Local\Temp\ipykernel_24028\649327763.py:6: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
      movies_metadata=pd.read_csv(r"C:\Users\lenovo\OneDrive\Documents\M2\FOND BIG DATA\movies_dataset\movies_metadata.csv")
    

Nous mettons en place une fonction permettant de transformer du texte en token, c'est-à-dire séparer chaque mot du texte pour pour préparer les données à l’analyse automatique : nettoyage, vectorisation et entraînement du modèle Word2Vec.


```python
#Tokenization d'un corpus de texte

from nltk.tokenize import word_tokenize #importation des librairies
import nltk
nltk.download("punkt_tab") 
def preprocessor(x): #On définit une fonction
    tokens = word_tokenize(x) #on sépare chaque mot du texte
    for i in range(len(tokens)): 
        try:
            float(tok[i]) #on vérifie si le token est un chiffre
            tokens[i] = 'NUM' #on le remplace par "NUM"
        except:
            tokens[i] = tokens[i].lower() #sinon on transforme tout le texte en miniscule
    return tokens  #retourne une liste avec chaque mot du texte


```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\lenovo\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

On va se concentrer sur la rubrique des "overview" qui correspond aux critiques émises sur chaque film de la base de données.


```python
movies_metadata["overview"]
```




    0        Led by Woody, Andy's toys live happily in his ...
    1        When siblings Judy and Peter discover an encha...
    2        A family wedding reignites the ancient feud be...
    3        Cheated on, mistreated and stepped on, the wom...
    4        Just when George Banks has recovered from his ...
                                   ...                        
    45461          Rising and falling between a man and woman.
    45462    An artist struggles to finish his work while a...
    45463    When one of her hits goes wrong, a professiona...
    45464    In a small town live two brothers, one a minis...
    45465    50 years after decriminalisation of homosexual...
    Name: overview, Length: 45466, dtype: object



On crée ensuite une colonne avec les critiques remplacés par des tokens.


```python
#Tokenisation des critiques de film 
import swifter 
movies_metadata['overview_clean'] = movies_metadata['overview'].fillna('').astype(str).swifter.apply(preprocessor) 
print(movies_metadata['overview_clean'].head())

```


    Pandas Apply:   0%|          | 0/45466 [00:00<?, ?it/s]


    0    [led, by, woody, ,, andy, 's, toys, live, happ...
    1    [when, siblings, judy, and, peter, discover, a...
    2    [a, family, wedding, reignites, the, ancient, ...
    3    [cheated, on, ,, mistreated, and, stepped, on,...
    4    [just, when, george, banks, has, recovered, fr...
    Name: overview_clean, dtype: object
    


```python
#Création de liste de tokens
sentences = movies_metadata.overview_clean.array 

```


```python
#Word2 Vec
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, min_count=3,window=5, workers=4) #calcul des vecteurs des tokens

model.train(sentences, epochs=15, total_examples=len(sentences)) #entrainement du modèle
```




    (193816225, 278301700)



`Vector_size = 100` signifie que chaque mot est représenté par un vecteur de 100 dimensions.

`Window = 5` signifie que le contexte se limite à 5 mots autour du mot cible.

`Workers = 4` signifie qu'on utilise 4 coeurs pour le traitement parallèle.

`Epochs = 15` signifie que le modèle va passer 15 fois en revue le corpus de texte. Cela améliore la qualité de l'apprentissage.

`Min_count = 3`signifie qu'on ne prend en compte que les mots qui apparaissent au moins 3 fois.


```python
# On peut prédire les mots les plus similaires à un mot donné
print(model.wv.most_similar("oscar", topn=10))

```

    [('award', 0.6227117776870728), ('nomination', 0.5970638990402222), ('nominee', 0.5693602561950684), ('emmy', 0.5425897836685181), ('award®', 0.5340083837509155), ('awards', 0.5217108726501465), ('watkins', 0.48955485224723816), ('multi-award', 0.4848301112651825), ('academy', 0.47046399116516113), ('faithfully', 0.46637263894081116)]
    


```python
#Test de similarité sémantique
print("Similarité entre 'action' et 'blockbuster' :", model.wv.similarity("action", "blockbuster"))
print("Similarité entre 'disney' et 'animated' :", model.wv.similarity("disney", "animated"))
print("Similarité entre 'action' et 'calm' :", model.wv.similarity("action", "calm"))




```

    Similarité entre 'action' et 'blockbuster' : 0.41067892
    Similarité entre 'disney' et 'animated' : 0.50679016
    Similarité entre 'action' et 'calm' : -0.10567398
    

<u>Interprétation du test de similarité:</u>

plus de 0.4 = mots très similaires 

entre 0.2 et 0.4 = mots modérément liés 

entre 0 et 0.2 = mots peu liés/différents 

négatif = mots opposés


```python
#Analyse vectorielle
print(model.wv.most_similar(positive=['horror', 'comedy'], negative=['drama']))
```

    [('spoof', 0.49941232800483704), ('all-new', 0.4678283631801605), ('sci-fi', 0.4596366882324219), ('genre', 0.4435296654701233), ('troma', 0.44133034348487854), ('snuff', 0.4343663156032562), ('immortalized', 0.43173038959503174), ('stand-up', 0.42355144023895264), ('anthology', 0.4218834936618805), ('skits', 0.4208984971046448)]
    

<u>Analyse vectorielle :</u> on souhaite rechercher un film d'horreur("horror") du registre comique("comedy") sans le coté dramatique, le 1er mot le plus similaire est parodie("spoof").


```python
# Taux de couverture du vocabulaire
vocab_coverage = sum(1 for tokens in sentences[:1000] if any(w in model.wv for w in tokens)) / 1000
print(f" Couverture vocabulaire : {vocab_coverage:.2%}")
```

     Couverture vocabulaire : 98.80%
    

 <u>Taux de couverture du vocabulaire :</u> mesure la proportion de phrases contenant au moins 1 mot connu du modèle
