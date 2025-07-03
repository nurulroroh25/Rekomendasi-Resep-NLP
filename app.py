# === KONFIGURASI AWAL & IMPORT ===
import streamlit as st
st.set_page_config(page_title="Rekomendasi Resep", page_icon="🍽️", layout="wide")

import pickle
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# === STYLING TAMBAHAN UI ===
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #fff8f0;
    }
    .stButton>button {
        background-color: #f63366;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4b89;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #f63366;
        padding: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# === LOAD MODEL NER IndoBERT ===
@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForTokenClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=10)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_pipeline = load_ner_model()

# === CLEANING DATASET RESEP ===
def clean_recipe_ingredients_column(df):
    df['Ingredients'] = df['Ingredients'].str.lower()
    df['Ingredients'] = df['Ingredients'].replace({
        r'\bdaging ayam\b': 'daging_ayam',
        r'\btelur ayam\b': 'telur_ayam',
        r'\bsosis ayam\b': 'sosis_ayam',
        r'\btelor\b': 'telur',
        r'\bcabe\b': 'cabai',
        r'\bcabai merah\b': 'cabai',
        r'\bcabai rawit\b': 'cabai',
        r'\bdaging sapi\b': 'sapi'
    }, regex=True)
    return df

# === LOAD DATA DAN TFIDF MODEL ===
try:
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open('recipes_data.pkl', 'rb') as f:
        recipes_df = pickle.load(f)

    recipes_df['Title'] = recipes_df['Title'].str.replace('*', '', regex=False).str.title()
    recipes_df = clean_recipe_ingredients_column(recipes_df)
    tfidf_matrix = tfidf_vectorizer.transform(recipes_df['Ingredients'].fillna(''))
    ingredients_vocab = tfidf_vectorizer.get_feature_names_out()

except FileNotFoundError:
    st.error("❌ File model atau data tidak ditemukan.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi error saat load data: {e}")
    st.stop()

# === NORMALISASI & DETEKSI BAHAN MAKANAN ===
def normalize_ingredients(text):
    replacements = {
        "daging ayam": "ayam",
        "ayam fillet": "ayam",
        "ayam kampung": "ayam",
        "daging sapi": "sapi",
        "ikan lele": "lele",
        "ikan nila": "nila",
        "ikan tongkol": "tongkol",
        "telur ayam": "telur",
        "telor": "telur",
        "sosis ayam": "sosis",
        "sosis sapi": "sosis",
        "minyak goreng": "minyak",
        "air putih": "air",
        "air matang": "air"
    }
    text = text.lower()
    for phrase, simple in replacements.items():
        text = text.replace(phrase, simple)
    return text

def detect_ingredients(text, vocab):
    text = normalize_ingredients(text)

    stopwords = {
        "saya", "punya", "ada", "mau", "ingin", "bisa", "enaknya", "cuma", "hanya",
        "masak", "dimasak", "dimakan", "apa", "ya", "nih", "dong", "sama",
        "pakai", "pake", "guna", "gunakan", "memakai", "memasak", "dan", "dengan", "juga", "atau",
        "aja", "deh", "lah","tapi","ga", "gak", "enggak", "tak", "tidak", "jgn", "jangan", "pengen"
    }

    negative_phrases = [
        r'tanpa\s+(\w+)',
        r'tanpa\s+pakai\s+(\w+)',
        r'tanpa\s+pake\s+(\w+)',
        r'tidak\s+pakai\s+(\w+)',
        r'tidak\s+(\w+)',
        r'tak\s+pakai\s+(\w+)',
        r'tak\s+(\w+)',
        r'ga\s+pakai\s+(\w+)',
        r'ga\s+pake\s+(\w+)',
        r'ga\s+(\w+)',
        r'gak\s+pakai\s+(\w+)',
        r'gak\s+pake\s+(\w+)',
        r'gak\s+(\w+)',
        r'jangan\s+pakai\s+(\w+)',
        r'jangan\s+pake\s+(\w+)',
        r'jgn\s+pake\s+(\w+)'
        r'jgn\s+pakai\s+(\w+)'
    ]

    negative_ingredients = set()
    for pattern in negative_phrases:
        matches = re.findall(pattern, text)
        for m in matches:
            negative_ingredients.add(m.strip().lower())

    # 1. Deteksi dengan NER
    entities = ner_pipeline(text)
    ingredients = [
        ent['word'].lower()
        for ent in entities
        if ent['entity_group'] in ['MISC', 'ORG']
    ]

    # 2. Fallback: jika NER kosong, pakai tokenisasi biasa
    if not ingredients:
        tokens = re.findall(r'\b\w+\b', text.lower())
        ingredients = [token for token in tokens if token in vocab]

    # 3. Bersihkan dari kata tidak penting
    noisy_words = {"pengen", "tanpa", "aja", "dong", "ya", "mau", "tapi"}
    filtered_ingredients = [
        i for i in ingredients
        if i in vocab and i not in stopwords and i not in noisy_words
    ]

    # 4. Filter bahan yang tidak termasuk bahan negatif
    detected = []
    for ing in filtered_ingredients:
        # Cek apakah ing ini adalah sinonim dari bahan negatif
        is_negative = any(
            ing in NEGATIVE_MAP.get(neg, [neg])
            for neg in negative_ingredients
        )
        if not is_negative:
            detected.append(ing)

    return list(set(detected)), negative_ingredients

# === CEK SINONIM BAHAN NEGATIF DI SETIAP RESEP ===
NEGATIVE_MAP = {
    "santan": ["santan", "air santan", "kelapa", "sari kelapa"],
    "cabai": ["cabai", "cabe", "sambal"],
    "keju": ["keju", "cheddar", "mozarella"],
    "gula": ["gula", "gula pasir", "manis"],
    "minyak": ["minyak", "minyak goreng", "minyak sayur", "minyak wijen", "menggoreng", "digoreng", "tumis", "menumis", "goreng"]
}

def contains_negative_ingredient(recipe_ingredients, recipe_steps, negative_ingredients):
    combined_text = f"{recipe_ingredients.lower()} {recipe_steps.lower()}"
    for neg in negative_ingredients:
        synonyms = NEGATIVE_MAP.get(neg, [neg])
        for syn in synonyms:
            if syn in combined_text:
                return True
    return False

# === REKOMENDASI RESEP BERDASARKAN KEMIRIPAN ===
def recommend_recipes(detected_ingredients, df, tfidf_matrix, tfidf_vectorizer, similarity_threshold=0.05, negative_ingredients=None):
    if not detected_ingredients:
        return []

    if negative_ingredients is None:
        negative_ingredients = set()

    user_vector = tfidf_vectorizer.transform([' '.join(detected_ingredients)])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()

    results = []
    for i, score in enumerate(cosine_sim):
        if score >= similarity_threshold:
            recipe_ingredients = df.iloc[i]['Ingredients'].lower()
            recipe_steps = df.iloc[i]['Steps'].lower()
            if contains_negative_ingredient(recipe_ingredients, recipe_steps, negative_ingredients):
                continue

            match_count = sum(ingredient in recipe_ingredients for ingredient in detected_ingredients)
            match_ratio = match_count / len(detected_ingredients)

            results.append({
                'title': df.iloc[i]['Title'],
                'ingredients': df.iloc[i]['Ingredients'],
                'steps': df.iloc[i]['Steps'],
                'similarity': score,
                'match_ratio': match_ratio
            })

    sorted_results = sorted(results, key=lambda x: (x['match_ratio'], x['similarity']), reverse=True)
    return sorted_results[:5]

# === UI UTAMA ===
st.markdown("""
    <div style="text-align:center; margin-top: -40px;">
        <h1 style="color:#f63366; font-size: 45px;">🍽️ Rekomendasi Resep Masakan</h1>
        <p style="font-size: 18px;">Yuk masak dari bahan yang kamu punya! Masukkan bahan-bahanmu di bawah 👇</p>
    </div>
""", unsafe_allow_html=True)

# === INPUT BAHAN ===
user_ingredients = st.text_input("Masukkan bahan masakan:", placeholder="Contoh: ayam, cabai, garam")
clicked = st.button("🔍 Cari Resep")

# === HASIL REKOMENDASI ===
if clicked:
    detected, negative_ingredients = detect_ingredients(user_ingredients, ingredients_vocab)
    st.markdown("<br>", unsafe_allow_html=True)

    if detected:
        st.success(f"✅ Bahan yang terdeteksi: {', '.join(detected)}")
        recommended = recommend_recipes(detected, recipes_df, tfidf_matrix, tfidf_vectorizer, negative_ingredients=negative_ingredients)

        if recommended:
            st.markdown("<h3 style='color:#f63366;'>🍲 Rekomendasi Resep:</h3>", unsafe_allow_html=True)
            for r in recommended:
                st.markdown(f"""
                    <div style="
                        background-color:#ffeef2;
                        padding: 20px;
                        border-radius: 18px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border-left: 6px solid #f63366;
                    ">
                        <h4 style="color:#d61c4e;">📌 {r['title']}</h4>
                        <b>📋 Bahan-bahan:</b>
                        <ul style="margin-left:15px;">
                            {''.join(f"<li>{i.strip()}</li>" for i in r['ingredients'].split('--') if i.strip())}
                        </ul>
                        <b>👨‍🍳 Langkah-langkah:</b>
                        <ol style="margin-left:15px;">
                            {''.join(f"<li>{step.strip()}</li>" for step in r['steps'].split('--') if step.strip())}
                        </ol>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("😕 Maaf, tidak ada resep yang cocok ditemukan.")
    else:
        st.warning("⚠️ Tidak ditemukan bahan dari input. Coba ketik ulang dengan bahan makanan yang lebih umum.")