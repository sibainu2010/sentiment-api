from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, re, math, os
import pymorphy3, nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
pipeline = joblib.load(MODEL_PATH)
morph = pymorphy3.MorphAnalyzer()
stopwords_ru = set(stopwords.words("russian"))
stopwords_ru.update({"это","что","был","была","было","весь","свой","который","очень","просто","все","ещё","уже","тоже","может","будет","есть","нет","да","для","как","так","меня","мне","себя","свою","свои","под","над","при","без","или","на","в","с","по","из","от","о","к","у","за","до","во","со","об","не","но","а","и","то","он","вас","теперь","них","вообще","этом","его","же","нас","вот","их","тут","еще","даже","когда","ну","только","поэтому","надо","ней","хотя","если","конечно","где","один","том","вы","совсем","чтобы","именно","ее","чем","этого","раз","много","ним","кто","всех","были","сразу","потому","видимо","им","почему","потом","нам","какой","про","общем","ли","ни","ничего","всё","всегда","вам","там","сказали","мастер","я","мы","они","бы","она","т","её","после","итоге","сами","пока","туда","сама"})
THRESHOLD = -0.197

def preprocess(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"[^а-яё\s]", "", text.lower())
    return " ".join(morph.parse(w)[0].normal_form for w in text.split() if w not in stopwords_ru and len(w) > 2)

class ReviewIn(BaseModel): text: str
class ReviewOut(BaseModel): label: str; emoji: str; confidence: int; clean_text: str

@app.get("/", response_class=HTMLResponse)
def index():
    BASE = os.path.dirname(__file__)
    with open(os.path.join(BASE, "index.html"), encoding="utf-8") as f: return f.read()

@app.post("/predict", response_model=ReviewOut)
def predict(req: ReviewIn):
    clean = preprocess(req.text)
    score = pipeline.decision_function([clean])[0]
    pos = score >= THRESHOLD
    return ReviewOut(label="Позитивный" if pos else "Негативный", emoji="😊" if pos else "😞", confidence=round(1/(1+math.exp(-abs(score)))*100), clean_text=clean or "—")

@app.get("/health")
def health(): return {"status": "ok"}
