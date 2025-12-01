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




```python
#Tokenisation des critiques de film 
import swifter 
movies_metadata['overview_clean'] = movies_metadata['overview'].fillna('').astype(str).swifter.apply(preprocessor) #On crée une colonne avec les critiques remplacés par des tokens
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

    <NumpyExtensionArray>
    [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ['led', 'by', 'woody', ',', 'andy', "'s", 'toys', 'live', 'happily', 'in', 'his', 'room', 'until', 'andy', "'s", 'birthday', 'brings', 'buzz', 'lightyear', 'onto', 'the', 'scene', '.', 'afraid', 'of', 'losing', 'his', 'place', 'in', 'andy', "'s", 'heart', ',', 'woody', 'plots', 'against', 'buzz', '.', 'but', 'when', 'circumstances', 'separate', 'buzz', 'and', 'woody', 'from', 'their', 'owner', ',', 'the', 'duo', 'eventually', 'learns', 'to', 'put', 'aside', 'their', 'differences', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ['when', 'siblings', 'judy', 'and', 'peter', 'discover', 'an', 'enchanted', 'board', 'game', 'that', 'opens', 'the', 'door', 'to', 'a', 'magical', 'world', ',', 'they', 'unwittingly', 'invite', 'alan', '--', 'an', 'adult', 'who', "'s", 'been', 'trapped', 'inside', 'the', 'game', 'for', '26', 'years', '--', 'into', 'their', 'living', 'room', '.', 'alan', "'s", 'only', 'hope', 'for', 'freedom', 'is', 'to', 'finish', 'the', 'game', ',', 'which', 'proves', 'risky', 'as', 'all', 'three', 'find', 'themselves', 'running', 'from', 'giant', 'rhinoceroses', ',', 'evil', 'monkeys', 'and', 'other', 'terrifying', 'creatures', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ['a', 'family', 'wedding', 'reignites', 'the', 'ancient', 'feud', 'between', 'next-door', 'neighbors', 'and', 'fishing', 'buddies', 'john', 'and', 'max', '.', 'meanwhile', ',', 'a', 'sultry', 'italian', 'divorcée', 'opens', 'a', 'restaurant', 'at', 'the', 'local', 'bait', 'shop', ',', 'alarming', 'the', 'locals', 'who', 'worry', 'she', "'ll", 'scare', 'the', 'fish', 'away', '.', 'but', 'she', "'s", 'less', 'interested', 'in', 'seafood', 'than', 'she', 'is', 'in', 'cooking', 'up', 'a', 'hot', 'time', 'with', 'max', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ['cheated', 'on', ',', 'mistreated', 'and', 'stepped', 'on', ',', 'the', 'women', 'are', 'holding', 'their', 'breath', ',', 'waiting', 'for', 'the', 'elusive', '``', 'good', 'man', "''", 'to', 'break', 'a', 'string', 'of', 'less-than-stellar', 'lovers', '.', 'friends', 'and', 'confidants', 'vannah', ',', 'bernie', ',', 'glo', 'and', 'robin', 'talk', 'it', 'all', 'out', ',', 'determined', 'to', 'find', 'a', 'better', 'way', 'to', 'breathe', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ['just', 'when', 'george', 'banks', 'has', 'recovered', 'from', 'his', 'daughter', "'s", 'wedding', ',', 'he', 'receives', 'the', 'news', 'that', 'she', "'s", 'pregnant', '...', 'and', 'that', 'george', "'s", 'wife', ',', 'nina', ',', 'is', 'expecting', 'too', '.', 'he', 'was', 'planning', 'on', 'selling', 'their', 'home', ',', 'but', 'that', "'s", 'a', 'plan', 'that', '--', 'like', 'george', '--', 'will', 'have', 'to', 'change', 'with', 'the', 'arrival', 'of', 'both', 'a', 'grandchild', 'and', 'a', 'kid', 'of', 'his', 'own', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ['obsessive', 'master', 'thief', ',', 'neil', 'mccauley', 'leads', 'a', 'top-notch', 'crew', 'on', 'various', 'insane', 'heists', 'throughout', 'los', 'angeles', 'while', 'a', 'mentally', 'unstable', 'detective', ',', 'vincent', 'hanna', 'pursues', 'him', 'without', 'rest', '.', 'each', 'man', 'recognizes', 'and', 'respects', 'the', 'ability', 'and', 'the', 'dedication', 'of', 'the', 'other', 'even', 'though', 'they', 'are', 'aware', 'their', 'cat-and-mouse', 'game', 'may', 'end', 'in', 'violence', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ['an', 'ugly', 'duckling', 'having', 'undergone', 'a', 'remarkable', 'change', ',', 'still', 'harbors', 'feelings', 'for', 'her', 'crush', ':', 'a', 'carefree', 'playboy', ',', 'but', 'not', 'before', 'his', 'business-focused', 'brother', 'has', 'something', 'to', 'say', 'about', 'it', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ['a', 'mischievous', 'young', 'boy', ',', 'tom', 'sawyer', ',', 'witnesses', 'a', 'murder', 'by', 'the', 'deadly', 'injun', 'joe', '.', 'tom', 'becomes', 'friends', 'with', 'huckleberry', 'finn', ',', 'a', 'boy', 'with', 'no', 'future', 'and', 'no', 'family', '.', 'tom', 'has', 'to', 'choose', 'between', 'honoring', 'a', 'friendship', 'or', 'honoring', 'an', 'oath', 'because', 'the', 'town', 'alcoholic', 'is', 'accused', 'of', 'the', 'murder', '.', 'tom', 'and', 'huck', 'go', 'through', 'several', 'adventures', 'trying', 'to', 'retrieve', 'evidence', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                   ['international', 'action', 'superstar', 'jean', 'claude', 'van', 'damme', 'teams', 'with', 'powers', 'boothe', 'in', 'a', 'tension-packed', ',', 'suspense', 'thriller', ',', 'set', 'against', 'the', 'back-drop', 'of', 'a', 'stanley', 'cup', 'game.van', 'damme', 'portrays', 'a', 'father', 'whose', 'daughter', 'is', 'suddenly', 'taken', 'during', 'a', 'championship', 'hockey', 'game', '.', 'with', 'the', 'captors', 'demanding', 'a', 'billion', 'dollars', 'by', 'game', "'s", 'end', ',', 'van', 'damme', 'frantically', 'sets', 'a', 'plan', 'in', 'motion', 'to', 'rescue', 'his', 'daughter', 'and', 'abort', 'an', 'impending', 'explosion', 'before', 'the', 'final', 'buzzer', '...'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ['james', 'bond', 'must', 'unmask', 'the', 'mysterious', 'head', 'of', 'the', 'janus', 'syndicate', 'and', 'prevent', 'the', 'leader', 'from', 'utilizing', 'the', 'goldeneye', 'weapons', 'system', 'to', 'inflict', 'devastating', 'revenge', 'on', 'britain', '.'],
     ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ['an', 'unsuccessful', 'sculptor', 'saves', 'a', 'madman', 'named', '``', 'the', 'creeper', "''", 'from', 'drowning', '.', 'seeing', 'an', 'opportunity', 'for', 'revenge', ',', 'he', 'tricks', 'the', 'psycho', 'into', 'murdering', 'his', 'critics', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ['in', 'this', 'true-crime', 'documentary', ',', 'we', 'delve', 'into', 'the', 'murder', 'spree', 'that', 'was', 'the', 'inspiration', 'for', 'joe', 'berlinger', "'s", '``', 'book', 'of', 'shadows', ':', 'blair', 'witch', '2', "''", '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ['a', 'film', 'archivist', 'revisits', 'the', 'story', 'of', 'rustin', 'parr', ',', 'a', 'hermit', 'thought', 'to', 'have', 'murdered', 'seven', 'children', 'while', 'under', 'the', 'possession', 'of', 'the', 'blair', 'witch', '.'],
                                                                                                                                                                                                                             ['it', "'s", 'the', 'year', '3000', 'ad', '.', 'the', 'world', "'s", 'most', 'dangerous', 'women', 'are', 'banished', 'to', 'a', 'remote', 'asteroid', '45', 'million', 'light', 'years', 'from', 'earth', '.', 'kira', 'murphy', 'does', "n't", 'belong', ';', 'wrongfully', 'accused', 'of', 'a', 'crime', 'she', 'did', 'not', 'commit', ',', 'she', "'s", 'thrown', 'in', 'this', 'interplanetary', 'prison', 'and', 'left', 'to', 'her', 'own', 'defenses', '.', 'but', 'kira', "'s", 'a', 'fighter', ',', 'and', 'soon', 'she', 'finds', 'herself', 'in', 'the', 'middle', 'of', 'a', 'female', 'gang', 'war', ';', 'where', 'everyone', 'wants', 'a', 'piece', 'of', 'the', 'action', '...', 'and', 'a', 'piece', 'of', 'her', '!', '``', 'caged', 'heat', '3000', "''", 'takes', 'the', 'women-in-prison', 'genre', 'to', 'a', 'whole', 'new', 'level', '...', 'and', 'a', 'whole', 'new', 'galaxy', '!'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ['yet', 'another', 'version', 'of', 'the', 'classic', 'epic', ',', 'with', 'enough', 'variation', 'to', 'make', 'it', 'interesting', '.', 'the', 'story', 'is', 'the', 'same', ',', 'but', 'some', 'of', 'the', 'characters', 'are', 'quite', 'different', 'from', 'the', 'usual', ',', 'in', 'particular', 'uma', 'thurman', "'s", 'very', 'special', 'maid', 'marian', '.', 'the', 'photography', 'is', 'also', 'great', ',', 'giving', 'the', 'story', 'a', 'somewhat', 'darker', 'tone', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ['rising', 'and', 'falling', 'between', 'a', 'man', 'and', 'woman', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ['an', 'artist', 'struggles', 'to', 'finish', 'his', 'work', 'while', 'a', 'storyline', 'about', 'a', 'cult', 'plays', 'in', 'his', 'head', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ['when', 'one', 'of', 'her', 'hits', 'goes', 'wrong', ',', 'a', 'professional', 'assassin', 'ends', 'up', 'with', 'a', 'suitcase', 'full', 'of', 'a', 'million', 'dollars', 'belonging', 'to', 'a', 'mob', 'boss', '...'],
     ['in', 'a', 'small', 'town', 'live', 'two', 'brothers', ',', 'one', 'a', 'minister', 'and', 'the', 'other', 'one', 'a', 'hunchback', 'painter', 'of', 'the', 'chapel', 'who', 'lives', 'with', 'his', 'wife', '.', 'one', 'dreadful', 'and', 'stormy', 'night', ',', 'a', 'stranger', 'knocks', 'at', 'the', 'door', 'asking', 'for', 'shelter', '.', 'the', 'stranger', 'talks', 'about', 'all', 'the', 'good', 'things', 'of', 'the', 'earthly', 'life', 'the', 'minister', 'is', 'missing', 'because', 'of', 'his', 'puritanical', 'faith', '.', 'the', 'minister', 'comes', 'to', 'accept', 'the', 'stranger', "'s", 'viewpoint', 'but', 'it', 'is', 'others', 'who', 'will', 'pay', 'the', 'consequences', 'because', 'the', 'minister', 'will', 'discover', 'the', 'human', 'pleasures', 'thanks', 'to', ',', 'ehem', ',', 'his', 'sister-', 'in', '-law…', 'the', 'tormented', 'minister', 'and', 'his', 'cuckolded', 'brother', 'will', 'die', 'in', 'a', 'strange', 'accident', 'in', 'the', 'chapel', 'and', 'later', 'an', 'infant', 'will', 'be', 'born', 'from', 'the', 'minister', "'s", 'adulterous', 'relationship', '.'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ['50', 'years', 'after', 'decriminalisation', 'of', 'homosexuality', 'in', 'the', 'uk', ',', 'director', 'daisy', 'asquith', 'mines', 'the', 'jewels', 'of', 'the', 'bfi', 'archive', 'to', 'take', 'us', 'into', 'the', 'relationships', ',', 'desires', ',', 'fears', 'and', 'expressions', 'of', 'gay', 'men', 'and', 'women', 'in', 'the', '20th', 'century', '.']]
    Length: 45466, dtype: object
    


```python
#Word2 Vec
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, workers=4) #calcul des vecteurs des tokens

model.train(sentences, epochs=100, total_examples=len(sentences)) #entrainement du modèle
```




    (190133181, 278301700)




```python
# On peut prédire les mots les plus similaires à un mot donné
print(model.wv.most_similar("french", topn=10))

```

    [('russian', 0.6964828372001648), ('spanish', 0.6715471148490906), ('british', 0.6626721620559692), ('german', 0.6563111543655396), ('japanese', 0.6459540128707886), ('italian', 0.6031754612922668), ('polish', 0.5751704573631287), ('hungarian', 0.5727871060371399), ('swedish', 0.562312126159668), ('czech', 0.5544315576553345)]
    


```python
#Teste de similarité sémantique
print(model.wv.similarity("action", "blockbuster"))
print(model.wv.similarity("disney", "animated"))
print(model.wv.similarity("action", "calm"))




```

    0.44138384
    0.50533974
    -0.12833318
    

<u>Interprétation du test de similarité:</u>

entre 0.4 et 1.0 = mots très similaires 

entre 0.2 et 0.4 = mots modérément liés 

entre 0 et 0.2 = mots peu liés/différents 

négatif = mots opposés


```python
#Analyse vectorielle
print(model.wv.most_similar(positive=['horror', 'comedy'], negative=['drama']))
```

    [('spoof', 0.49941232800483704), ('all-new', 0.4678283631801605), ('sci-fi', 0.4596366882324219), ('genre', 0.4435296654701233), ('troma', 0.44133034348487854), ('snuff', 0.4343663156032562), ('immortalized', 0.43173038959503174), ('stand-up', 0.42355144023895264), ('anthology', 0.4218834936618805), ('skits', 0.4208984971046448)]
    

Analyse vectorielle : on souhaite rechercher un film d'horreur("horror") du registre comique("comedy") sans le coté dramatique, le 1er mot le plus similaire est parodie("spoof").


```python
# Taux de couverture du vocabulaire
vocab_coverage = sum(1 for tokens in sentences[:1000] if any(w in model.wv for w in tokens)) / 1000
print(f" Couverture vocabulaire : {vocab_coverage:.2%}")
```

     Couverture vocabulaire : 98.80%
    

 Taux de couverture du vocabulaire : mesure la proportion de phrases contenant au moins 1 mot connu du modèle


```python

```
