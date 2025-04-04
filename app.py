from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, os, requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# Flask uygulaması
app = Flask(__name__)

# CORS ayarları (sadece batuhandurmaz.com'dan gelen istekleri kabul et)
CORS(app, origins=["https://www.batuhandurmaz.com"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])

# OpenAI API anahtarı
openai.api_key = os.getenv("OPENAI_API_KEY")

# HTML'den metin ayıklama
def extract_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception:
        return ""

# Embedding oluştur
def get_embedding(text):
    text = text[:4000]  # Uzun metinlerin token limitine takılmaması için kırp
    res = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

# İçerik boşluğu analizi + anahtar kelime odaklı değerlendirme
def suggest_gap(my_text, comp_text, keyword):
    prompt = f"""
Hedef Anahtar Kelime: {keyword}

Benim İçeriğim:
{my_text[:2000]}

Rakip İçerik:
{comp_text[:2000]}

Yukarıdaki içerikleri karşılaştır.
- Özellikle hedef anahtar kelimeye göre içerik kalitesini değerlendir.
- Benim içeriğimde eksik olan ama rakibin içeriğinde bulunan başlıklar, açıklamalar veya detayları listele.
- SEO açısından hangi boşlukları kapatmalıyım?
"""
    res = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# Anahtar kelime için öneriler, n-gramlar ve entity tahmini
def expand_keyword_insights(keyword):
    prompt = f"""
"Aşağıdaki anahtar kelime için SEO odaklı öneriler üret:

1. {keyword} kelimesiyle ilişkili long-tail keyword (n-gram) önerileri.
2. Bu kelimeyle bağlantılı olabilecek entity'leri (markalar, kişiler, kavramlar).
3. Kullanıcı bu kelimeyi yazarken ne gibi arama niyetleri olabilir? (intent analizi).

Kısa maddeler halinde yaz."
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# API endpoint
@app.route("/api/seo-analyze", methods=["POST"])
def seo_analyze():
    data = request.json
    keyword = data.get("keyword", "")
    my_text = extract_text(data["my_url"])
    my_embedding = get_embedding(my_text)

    results = []
    for url in data["competitors"]:
        try:
            comp_text = extract_text(url)
            comp_embedding = get_embedding(comp_text)
            sim = cosine_similarity([my_embedding], [comp_embedding])[0][0]
            suggestion = suggest_gap(my_text, comp_text, keyword) if sim < 0.85 else ""
            results.append({
                "url": url,
                "similarity": round(sim, 3),
                "suggestion": suggestion
            })
        except Exception as e:
            results.append({
                "url": url,
                "similarity": "Hata",
                "suggestion": str(e)
            })

    # Anahtar kelime önerileri
    keyword_expansion = ""
    try:
        keyword_expansion = expand_keyword_insights(keyword)
    except Exception as e:
        keyword_expansion = f"Kelime önerileri alınamadı: {str(e)}"

    return jsonify({
        "analysis": results,
        "keyword_expansion": keyword_expansion
    })

# Railway port ayarı (host: 0.0.0.0, port: dinamik)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
