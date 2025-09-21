# 🎬 **Sentence Transformers vs TF-IDF + KNN per Recommendation Engine**

Questo documento spiega il funzionamento dei due approcci implementati nel progetto per il sistema di raccomandazione di film.

## **1. Sentence Transformers (Approccio basato su Deep Learning)**

### **Come Funziona:**
```
Film Overview → Sentence Transformer → Dense Embeddings (384D) → Cosine Similarity → Raccomandazioni
```

**Passi principali:**
1. **Modello Pre-addestrato**: Usa `all-MiniLM-L6-v2`, un modello BERT ottimizzato per similarity semantica
2. **Encoding Semantico**: Converte ogni overview del film in un vettore denso di 384 dimensioni che cattura il significato semantico
3. **Ricerca per Similarità**: Calcola la cosine similarity tra gli embeddings per trovare film simili
4. **Comprensione Contestuale**: Capisce sinonimi, contesto e relazioni semantiche complesse

### **Implementazione nel Progetto:**

#### **Training (model.ipynb):**
```python
# Caricamento del modello pre-addestrato
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encoding degli overview dei film
movie_embeddings = model.encode(
    movies["overview"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

# Salvataggio degli embeddings
joblib.dump(movie_embeddings, "pickle_model/movie_embeddings.pkl")
```

#### **Inference (app_sent_transf.py):**
```python
def recommend(movies, embeddings, movie_name, top_k=15):
    # Trova l'indice del film
    idx = movies[movies["names"].str.lower() == movie_name.lower()].index[0]
    query_embedding = embeddings[idx]
    
    # Calcola cosine similarity
    scores = np.dot(embeddings_np, query_embedding) / (
        norm(embeddings_np, axis=1) * norm(query_embedding)
    )
    
    # Restituisce i top film simili
    top_indices = np.argsort(scores)[::-1][1:top_k+1]
    return movies.iloc[top_indices]["names"].tolist()
```

### **Vantaggi:**
- ✅ **Comprensione Semantica Profonda**: Capisce che "automobile" e "car" sono simili
- ✅ **Gestione Sinonimi**: Riconosce concetti correlati anche con parole diverse
- ✅ **Qualità delle Raccomandazioni**: Più accurate grazie alla comprensione del contesto
- ✅ **Robustezza**: Funziona bene anche con testi brevi o con errori

### **Svantaggi:**
- ❌ **Computazionalmente Intensivo**: Richiede più risorse per l'encoding
- ❌ **Modello Pre-addestrato**: Dipende dalla qualità del training del modello
- ❌ **Memoria**: Gli embeddings occupano più spazio (384 dimensioni per film)

---

## **2. TF-IDF + KNN (Approccio Tradizionale)**

### **Come Funziona:**
```
Film Overview → TF-IDF Vectorization → Sparse Vectors → KNN Cosine Distance → Raccomandazioni
```

**Passi principali:**
1. **TF-IDF Vectorization**: Converte i testi in vettori sparsi basati su frequenza delle parole
   - **TF (Term Frequency)**: Quanto spesso appare una parola nel documento
   - **IDF (Inverse Document Frequency)**: Quanto è rara una parola nel corpus
2. **KNN (K-Nearest Neighbors)**: Trova i K film più vicini usando distanza coseno
3. **Matching Lessicale**: Si basa sulla sovrapposizione di parole specifiche

### **Implementazione nel Progetto:**

#### **Training (model.ipynb):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Creazione della matrice TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(movies["overview"])

# Training del modello KNN
knn = NearestNeighbors(n_neighbors=16, metric="cosine", algorithm="brute")
knn.fit(tfidf_matrix)

# Salvataggio dei modelli
joblib.dump(knn, "pickle_model/knn_model.pkl")
joblib.dump(tfidf, "pickle_model/tfidf_vectorizer.pkl")
```

#### **Inference (app_tfidf.py):**
```python
def recommend(movie_name, knn_model, tfidf_vectorizer, movies_df, n_recommendations=10):
    # Trova l'indice del film
    movie_index = movies_df[movies_df["names"] == movie_name].index[0]
    
    # Ottieni il vettore TF-IDF del film
    movie_vector = tfidf_vectorizer.transform([movies_df.iloc[movie_index]["overview"]])
    
    # Trova film simili usando KNN
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)
    
    # Restituisce i film raccomandati (esclude il film stesso)
    recommended_movies = []
    for idx in indices[0][1:]:
        recommended_movies.append(movies_df.iloc[idx]["names"])
    
    return recommended_movies
```

### **Vantaggi:**
- ✅ **Veloce e Leggero**: Computazionalmente efficiente
- ✅ **Interpretabile**: Puoi vedere esattamente quali parole influenzano le raccomandazioni
- ✅ **Controllo Preciso**: Facile da personalizzare e mettere a punto
- ✅ **Memoria Efficiente**: Vettori sparsi occupano meno spazio

### **Svantaggi:**
- ❌ **Solo Matching Lessicale**: Non capisce sinonimi o contesto semantico
- ❌ **Sensibile a Preprocessing**: Dipende molto dalla pulizia del testo
- ❌ **Vocabulario Limitato**: Funziona solo con parole esatte del training set
- ❌ **Curse of Dimensionality**: Con grandi vocabulari diventa meno efficace

---

## **3. Confronto Pratico**

| Aspetto | Sentence Transformers | TF-IDF + KNN |
|---------|----------------------|---------------|
| **Qualità Raccomandazioni** | ⭐⭐⭐⭐⭐ Eccellente | ⭐⭐⭐ Buona |
| **Velocità** | ⭐⭐⭐ Media | ⭐⭐⭐⭐⭐ Velocissima |
| **Consumo Memoria** | ⭐⭐ Alto | ⭐⭐⭐⭐ Basso |
| **Interpretabilità** | ⭐⭐ Blackbox | ⭐⭐⭐⭐⭐ Trasparente |
| **Gestione Sinonimi** | ⭐⭐⭐⭐⭐ Eccellente | ⭐ Limitata |

---

## **4. Esempio Pratico**

### **Query**: "Un film su auto da corsa"

#### **Sentence Transformers** troverebbe:
- "Fast & Furious" (capisce "racing cars")
- "Rush" (capisce il contesto di corse)
- "Ford v Ferrari" (associa contesto automobilistico)

#### **TF-IDF + KNN** troverebbe solo film che contengono esattamente:
- "car", "auto", "racing" nel testo
- Perderebbe film con sinonimi come "vehicle", "automobile", "motorsport"

---

## **5. Quando Usare Quale Approccio**

### **Usa Sentence Transformers quando:**
- Hai risorse computazionali sufficienti
- Vuoi la massima qualità delle raccomandazioni
- I tuoi utenti usano query semanticamente complesse
- Hai un dataset multilingue o con sinonimi

### **Usa TF-IDF + KNN quando:**
- Hai vincoli di performance stretti
- Vuoi un sistema interpretabile e controllabile
- Hai un dataset con terminologia molto specifica
- Vuoi un sistema semplice da debuggare e mantenere

---

## **6. Architettura del Progetto**

```
📁 Recommendation-engine-NLP/
├── 📁 Sentence-Transformer/
│   ├── app_sent_transf.py      # Applicazione Streamlit
│   ├── model.ipynb             # Training del modello
│   ├── imdb_movies.csv         # Dataset
│   └── 📁 pickle_model/
│       ├── sentence_transformer_model.pkl
│       ├── movie_embeddings.pkl
│       └── movies_dataset.pkl
│
├── 📁 TFIDF-KNN/
│   ├── app_tfidf.py           # Applicazione Streamlit
│   ├── model.ipynb            # Training del modello
│   ├── imdb_movies.csv        # Dataset
│   └── 📁 pickle_model/
│       ├── knn_model.pkl
│       └── tfidf_vectorizer.pkl
│
└── 📁 TMDB-API/
    └── app_tmdb.py            # Integrazione con API TMDB
```

---

## **7. Conclusioni**

Entrambi gli approcci hanno i loro punti di forza e debolezza. La scelta dipende dai requisiti specifici del progetto:

- **Per applicazioni dove la qualità è prioritaria**: Sentence Transformers
- **Per applicazioni dove la velocità è critica**: TF-IDF + KNN
- **Per applicazioni ibride**: Considera l'uso di entrambi in parallelo

Il progetto implementa entrambe le soluzioni, permettendo un confronto diretto e la scelta dell'approccio più adatto al caso d'uso specifico.