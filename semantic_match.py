# semantic_match_full.py
"""
Complete Streamlit matrimony semantic-matching demo.
- Hybrid retrieval: BM25 + FAISS (semantic ANN)
- Hard filters: age, gender, caste, race, min-height, settled, distance
- Pairwise features + heuristic scoring (dynamic weights based on user preferences)
- Optional Cross-Encoder refinement (if installed + desired)
- Attractive UI with Lottie animation + CSS animations
- Embeddings cached to disk for faster restarts
"""

import os
import math
import random
import time
import json
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Try optional CrossEncoder
USE_CROSS_ENCODER = False  # flip to True if you installed 'sentence-transformers' fully and want CE
CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"

try:
    if USE_CROSS_ENCODER:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    else:
        cross_encoder = None
except Exception:
    cross_encoder = None
    USE_CROSS_ENCODER = False

# -------------------------
# App settings
# -------------------------
st.set_page_config(page_title="Matrimony Match — Full Demo", layout="wide", page_icon="💞")
st.markdown("<style>body{background:linear-gradient(180deg,#fff,#fbfdff);}</style>", unsafe_allow_html=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
CACHE_DIR = "match_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_PROFILES = 800   # reduce to 200-400 for low-RAM machines
CANDIDATE_POOL = 350
TOP_K = 10
RSEED = 42
random.seed(RSEED)
np.random.seed(RSEED)

# -------------------------
# Utilities
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    try:
        R = 6371.0
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except Exception:
        return 9999.0

# -------------------------
# Synthetic dataset builder (with extra fields & random "requirements")
# -------------------------
def create_synthetic_profiles(n=NUM_PROFILES):
    names = [f"User_{i}" for i in range(n)]
    occupations = ["Software Engineer","Doctor","Business","Teacher","Civil Engineer","Designer","Analyst","Manager"]
    educations = ["B.Tech","M.Tech","B.Sc","MBA","PhD"]
    hobbies_pool = ["reading","trekking","movies","cooking","travel","photography","music","sports","yoga","painting","gardening"]
    religions = ["Hindu","Muslim","Christian","Sikh"]
    cities = [
        (17.3850,78.4867,"Hyderabad"),
        (13.0827,80.2707,"Chennai"),
        (12.9716,77.5946,"Bangalore"),
        (19.0760,72.8777,"Mumbai"),
        (28.7041,77.1025,"Delhi")
    ]
    races = ["Fair","Wheatish","Dark"]
    wealth_levels = ["low","mid","high"]

    def random_requirement_for(gender):
        req_same_caste = random.random() < 0.25
        r = random.random()
        if r < 0.25:
            min_height_cm = 178
        elif r < 0.6:
            min_height_cm = 165
        else:
            min_height_cm = None
        req_race_choice = None
        if random.random() < 0.20:
            req_race_choice = random.sample(races, k=random.choice([1,2]))
        req_well_settled = random.random() < 0.25
        req_gender_pref = "Male" if gender=="Female" else "Female"
        return {
            "caste_required": bool(req_same_caste),
            "min_height_cm": min_height_cm,
            "required_race": req_race_choice,
            "require_well_settled": bool(req_well_settled),
            "gender_pref": req_gender_pref
        }

    rows = []
    for i in range(n):
        age = random.randint(22, 36)
        gender = random.choice(["Male","Female"])
        religion = random.choice(religions)
        lat0, lon0, city = random.choice(cities)
        lat = lat0 + random.uniform(-0.4, 0.4)
        lon = lon0 + random.uniform(-0.4, 0.4)
        caste = random.choice(["CasteA","CasteB","CasteC", None])
        education = random.choice(educations)
        occupation = random.choice(occupations)
        hobbies = random.sample(hobbies_pool, k=3)
        height_cm = random.randint(150, 190) if gender=="Male" else random.randint(145, 175)
        wealth = random.choices(wealth_levels, weights=[0.5,0.35,0.15])[0]
        well_settled = True if (wealth=="high" or occupation in ["Manager","Business","Doctor"]) else (random.random() < 0.18)
        race_color = random.choice(races)
        bio = f"{age}-year-old {occupation} based in {city}. Family oriented, likes {', '.join(hobbies)}. Values honesty."
        txt = bio + " Interests: " + ", ".join(hobbies)
        requirements = random_requirement_for(gender)
        rows.append({
            "id": i,
            "name": names[i],
            "age": age,
            "gender": gender,
            "religion": religion,
            "lat": lat,
            "lon": lon,
            "caste": caste,
            "education": education,
            "occupation": occupation,
            "bio": bio,
            "interests": ",".join(hobbies),
            "txt": txt,
            "response_rate": float(np.clip(np.random.beta(2,5), 0,1)),
            "views_30d": int(np.random.randint(0,300)),
            "height_cm": height_cm,
            "wealth_level": wealth,
            "well_settled": bool(well_settled),
            "race_color": race_color,
            "requirements": requirements
        })
    return pd.DataFrame(rows)

# -------------------------
# Build indices: BM25, embeddings, FAISS index (cache embeddings)
# -------------------------
@st.cache_resource
def build_indices(num_profiles=NUM_PROFILES):
    profiles = create_synthetic_profiles(num_profiles)
    docs = profiles["txt"].astype(str).tolist()
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer(EMBED_MODEL)
    emb_path = os.path.join(CACHE_DIR, "embeddings.npy")
    if os.path.exists(emb_path) and os.path.getsize(emb_path) > 0:
        embs = np.load(emb_path)
    else:
        embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        np.save(emb_path, embs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embs)
    return {"profiles": profiles, "bm25": bm25, "model": model, "index": index, "embs": embs}

PIPE = build_indices(NUM_PROFILES)
PROFILES = PIPE["profiles"]
BM25 = PIPE["bm25"]
MODEL = PIPE["model"]
INDEX = PIPE["index"]
EMBS = PIPE["embs"]
EMB_MAP = {int(PROFILES.iloc[i]['id']): EMBS[i] for i in range(len(PROFILES))}

# -------------------------
# Retrieval (hybrid BM25 + ANN)
# -------------------------
def hybrid_retrieve(query_text: str, top_k=200, bm25_k=200, bm25_weight=0.6, ann_weight=0.4) -> List[int]:
    tokenized_q = query_text.split()
    bm25_scores = BM25.get_scores(tokenized_q)
    bm25_idx = np.argsort(-bm25_scores)[:bm25_k]
    bm25_ids = [int(PROFILES.iloc[i]["id"]) for i in bm25_idx]
    q_emb = MODEL.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = INDEX.search(q_emb, top_k)
    ann_ids = [int(PROFILES.iloc[int(i)]["id"]) for i in I[0]]
    candidate_scores = {}
    for rank, pid in enumerate(bm25_ids):
        candidate_scores[pid] = candidate_scores.get(pid, 0.0) + bm25_weight * (1.0 / (1 + rank))
    for rank, pid in enumerate(ann_ids):
        candidate_scores[pid] = candidate_scores.get(pid, 0.0) + ann_weight * float(D[0][rank])
    sorted_pids = sorted(candidate_scores.items(), key=lambda kv: -kv[1])
    return [int(pid) for pid, s in sorted_pids[:top_k]]

# -------------------------
# Hard filters
# -------------------------
def hard_filter(candidate_ids: List[int],
                requester: Dict[str, Any],
                max_distance_km: float = 100.0,
                gender_pref: Optional[str] = None,
                max_age_diff: int = 10,
                caste_required: bool = False,
                min_height_cm: Optional[int] = None,
                required_race: Optional[List[str]] = None,
                require_well_settled: Optional[bool] = None,
                enforce_opposite_gender: bool = True) -> List[int]:
    df = PROFILES.set_index("id")
    out = []
    for pid in candidate_ids:
        if int(pid) == int(requester.get("id", -1)):
            continue
        row = df.loc[int(pid)]
        # enforce opposite gender
        if enforce_opposite_gender and requester.get("gender") is not None:
            if row.get("gender") == requester.get("gender"):
                continue
        if gender_pref and row['gender'] != gender_pref:
            continue
        # age filter
        if requester.get("age") is not None and row.get("age") is not None:
            if abs(row['age'] - requester.get("age", 30)) > max_age_diff:
                continue
        if caste_required and (row['caste'] != requester.get("caste")):
            continue
        if min_height_cm is not None and (row.get("height_cm", 0) < min_height_cm):
            continue
        if required_race and (row.get("race_color") not in required_race):
            continue
        if require_well_settled and not bool(row.get("well_settled", False)):
            continue
        d = haversine_km(requester.get("lat", 17.3850), requester.get("lon", 78.4867), float(row["lat"]), float(row["lon"]))
        if d > max_distance_km:
            continue
        # candidate's own requirements (mutual compatibility)
        cand_reqs = row.get("requirements", {}) or {}
        if cand_reqs.get("gender_pref") and requester.get("gender") and requester.get("gender") != cand_reqs.get("gender_pref"):
            continue
        if cand_reqs.get("caste_required", False):
            if requester.get("caste") is None or requester.get("caste") != row.get("caste"):
                continue
        req_min_h = cand_reqs.get("min_height_cm")
        if req_min_h is not None:
            if requester.get("height_cm") is None or requester.get("height_cm") < req_min_h:
                continue
        req_race = cand_reqs.get("required_race")
        if req_race and (requester.get("race_color") not in req_race):
            continue
        if cand_reqs.get("require_well_settled", False):
            if not bool(requester.get("well_settled", False)):
                continue
        out.append(int(pid))
    return out

# -------------------------
# Feature engineering for reranker
# -------------------------
def compute_pair_features(requester_profile: Dict[str, Any], candidate_profile: Dict[str, Any]) -> Dict[str, float]:
    # requester embedding
    if "_embedding" in requester_profile and requester_profile["_embedding"] is not None:
        emb_a = requester_profile["_embedding"]
    else:
        rid = int(requester_profile.get("id", -1))
        if rid in EMB_MAP:
            emb_a = EMB_MAP[rid]
        else:
            emb_a = MODEL.encode([requester_profile.get("txt", "")], convert_to_numpy=True, normalize_embeddings=True)[0]
            requester_profile["_embedding"] = emb_a
    # candidate embedding
    cid = int(candidate_profile.get("id", -1))
    emb_b = EMB_MAP.get(cid, MODEL.encode([candidate_profile.get("txt", "")], convert_to_numpy=True, normalize_embeddings=True)[0])

    cos = float(cosine_similarity([emb_a], [emb_b])[0][0])
    ia = set(str(requester_profile.get("interests", "")).split(",")) if requester_profile.get("interests") else set()
    ib = set(str(candidate_profile.get("interests", "")).split(",")) if candidate_profile.get("interests") else set()
    inter_overlap = len(ia.intersection(ib))

    caste_match = 1.0 if (requester_profile.get("caste") is not None and requester_profile.get("caste") == candidate_profile.get("caste")) else 0.0
    race_match = 1.0 if (requester_profile.get("race_color") is not None and requester_profile.get("race_color") == candidate_profile.get("race_color")) else 0.0
    settled_match = 1.0 if bool(candidate_profile.get("well_settled", False)) else 0.0
    try:
        height_diff = abs(float(requester_profile.get("height_cm", 170) or 170) - float(candidate_profile.get("height_cm", 170) or 170))
    except Exception:
        height_diff = 999.0

    feats = {
        "emb_cosine": cos,
        "inter_overlap": inter_overlap,
        "same_religion": 1.0 if requester_profile.get("religion") == candidate_profile.get("religion") else 0.0,
        "same_education": 1.0 if requester_profile.get("education") == candidate_profile.get("education") else 0.0,
        "same_occupation_type": 1.0 if requester_profile.get("occupation") == candidate_profile.get("occupation") else 0.0,
        "age_diff": abs(requester_profile.get("age", 0) - candidate_profile.get("age", 0)),
        "distance_km": haversine_km(requester_profile.get("lat", 17.3850), requester_profile.get("lon", 78.4867),
                                   candidate_profile.get("lat", 17.3850), candidate_profile.get("lon", 78.4867)),
        "candidate_response_rate": float(candidate_profile.get("response_rate", random.random())),
        "profile_views_last_30d": float(candidate_profile.get("views_30d", 0)),
        "caste_match": caste_match,
        "race_match": race_match,
        "settled_match": settled_match,
        "height_diff": height_diff,
        "height_compat_score": max(0.0, 1.0 - (height_diff / 30.0))
    }
    feats["age_compat_score"] = max(0.0, 1.0 - (feats["age_diff"] / 20.0))
    return feats

def features_to_vector(feature_dict: Dict[str, float]) -> np.ndarray:
    keys = ["emb_cosine","inter_overlap","same_religion","same_education","same_occupation_type",
            "age_diff","distance_km","candidate_response_rate","profile_views_last_30d","age_compat_score",
            "caste_match","race_match","settled_match","height_diff","height_compat_score"]
    return np.array([feature_dict.get(k, 0.0) for k in keys], dtype=float)

# -------------------------
# Dynamic weighting & scoring
# -------------------------
def compute_dynamic_weights(preferences: Dict[str, Any]) -> np.ndarray:
    """
    preferences: age_importance [0-1], height_importance [0-1], gender_age_preference: "partner_older"/"partner_younger"/None
    """
    base = np.array([3.0,0.6,0.8,0.5,0.5,-0.1,-0.01,1.2,0.001,1.0,1.5,0.6,1.2,-0.05,1.0], dtype=float)
    age_imp = float(preferences.get("age_importance", 0.0))
    height_imp = float(preferences.get("height_importance", 0.0))
    gender_age_pref = preferences.get("gender_age_preference", None)
    # amplify age-related weights proportional to importance
    base[9] *= (1.0 + 1.5 * age_imp)
    base[5] *= (1.0 + 0.6 * age_imp)
    # height influence
    base[14] *= (1.0 + 1.5 * height_imp)
    base[13] *= (1.0 + 0.2 * height_imp)
    # if user prefers partner older, encourage male_older (we don't have explicit male_older in vector here; handled later)
    if gender_age_pref == "partner_older":
        # increase age compat importance
        base[9] *= 1.2
        base[10] *= 1.1  # caste slightly more important if age matters
    return base

def heuristic_score(vec: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(vec, weights))

# -------------------------
# Explainability & icebreaker
# -------------------------
def explainability_labels(feats: Dict[str, float]) -> List[str]:
    labels = []
    if feats["inter_overlap"] >= 2:
        labels.append("High interests match")
    if feats["age_compat_score"] > 0.7 and feats["distance_km"] < 50:
        labels.append("Compatible lifestyle & location")
    if feats.get("caste_match", 0.0) == 1.0:
        labels.append("Same caste")
    if feats.get("race_match", 0.0) == 1.0:
        labels.append("Race/color match")
    if feats.get("settled_match", 0.0) == 1.0:
        labels.append("Well-settled")
    if feats.get("height_compat_score", 0.0) > 0.8:
        labels.append("Height match")
    if feats.get("candidate_response_rate", 0.0) > 0.6:
        labels.append("Good responder")
    if len(labels) == 0:
        labels.append("Potential match")
    return labels

def suggest_icebreaker(req: Dict[str, Any], cand: Dict[str, Any], feats: Dict[str, float]) -> str:
    ia = set(str(req.get("interests","")).split(",")) if req.get("interests") else set()
    ib = set(str(cand.get("interests","")).split(",")) if cand.get("interests") else set()
    common = ia.intersection(ib)
    if common:
        topic = list(common)[0]
        return f"Hey {cand.get('name')}, I also love {topic} — what's your favorite {topic} memory?"
    if feats.get("settled_match", 0.0) == 1.0:
        return f"Hi {cand.get('name')}, I see you're well-settled — what's a typical workday like for you?"
    return f"Hi {cand.get('name')}, your profile stood out — would you like to chat?"

# -------------------------
# End-to-end matching function
# -------------------------
def run_match_for_user_profile(user_profile: Dict[str, Any], preferences: Dict[str, Any], candidate_pool: int = CANDIDATE_POOL, top_k: int = TOP_K, relax_if_empty: bool = True):
    # Normalize requester
    req = dict(user_profile)
    req.setdefault("id", -1)
    req.setdefault("name", req.get("name", "You"))
    req.setdefault("age", int(req.get("age", 28)) if req.get("age") is not None else 28)
    req.setdefault("gender", req.get("gender", None))
    req.setdefault("lat", float(req.get("lat", 17.3850)))
    req.setdefault("lon", float(req.get("lon", 78.4867)))
    req.setdefault("height_cm", req.get("height_cm", None))
    req.setdefault("interests", req.get("interests", ""))
    if isinstance(req["interests"], list):
        req["interests"] = ",".join(req["interests"])
    if "txt" not in req or not req["txt"]:
        req["txt"] = (req.get("bio","") or "") + " Interests: " + (req.get("interests","") or "")
    # compute requester embedding
    if "_embedding" not in req or req["_embedding"] is None:
        try:
            req["_embedding"] = MODEL.encode([req["txt"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        except Exception:
            req["_embedding"] = None

    # candidate generation
    hybrid_ids = hybrid_retrieve(req["txt"], top_k=candidate_pool, bm25_k=200)
    hybrid_ids = [int(x) for x in hybrid_ids if int(x) != int(req.get("id", -1))]

    # hard filtering using preferences
    filtered = hard_filter(hybrid_ids, req,
                           max_distance_km=preferences.get("max_distance_km", 150.0),
                           gender_pref=preferences.get("gender_pref", None),
                           max_age_diff=preferences.get("max_age_diff", 8),
                           caste_required=preferences.get("caste_required", False),
                           min_height_cm=preferences.get("min_height_cm", None),
                           required_race=preferences.get("required_race", None),
                           require_well_settled=preferences.get("require_well_settled", None),
                           enforce_opposite_gender=True)
    # relax if none
    if len(filtered) == 0 and relax_if_empty:
        filtered = hard_filter(hybrid_ids, req,
                               max_distance_km=preferences.get("max_distance_km", 200.0),
                               gender_pref=preferences.get("gender_pref", None),
                               max_age_diff=preferences.get("max_age_diff", 20),
                               caste_required=False,
                               min_height_cm=None,
                               required_race=None,
                               require_well_settled=None,
                               enforce_opposite_gender=True)

    if len(filtered) == 0:
        return pd.DataFrame([])

    # compute features & score
    dyn_weights = compute_dynamic_weights(preferences or {})
    candidate_rows = []
    feature_vectors = []
    for pid in filtered:
        cand = PROFILES[PROFILES['id'] == pid].iloc[0].to_dict()
        cand['id'] = int(cand['id'])
        feats = compute_pair_features(req, cand)
        vec = features_to_vector(feats)
        candidate_rows.append((pid, cand, feats))
        feature_vectors.append(vec)
    feature_matrix = np.vstack(feature_vectors) if len(feature_vectors) > 0 else np.zeros((0, len(dyn_weights)))
    scores = []
    for i in range(len(candidate_rows)):
        vec = feature_matrix[i]
        s = heuristic_score(vec, dyn_weights)
        scores.append(s)

    # Cross-encoder refinement for top-K if available (optional)
    ranked_idx = np.argsort(-np.array(scores))
    if cross_encoder is not None:
        top_for_ce = ranked_idx[:min(len(ranked_idx), 20)]  # small set for CE
        pairs = []
        mapping = []
        for pos in top_for_ce:
            pid, cand, feats = candidate_rows[int(pos)]
            pairs.append((req["txt"], cand["txt"]))
            mapping.append(pos)
        try:
            ce_scores = cross_encoder.predict(pairs)
            # integrate CE by replacing the scores for these positions with CE*scale + original*scale
            for j, pos in enumerate(mapping):
                scores[int(pos)] = 0.45 * scores[int(pos)] + 0.55 * float(ce_scores[j])
        except Exception:
            pass  # if CE fails, ignore

    # final ranking
    final_idx = np.argsort(-np.array(scores))[:top_k]
    final = []
    for rank_pos, idx in enumerate(final_idx):
        pid, cand, feats = candidate_rows[int(idx)]
        score = float(scores[int(idx)])
        labels = explainability_labels(feats)
        ice = suggest_icebreaker(req, cand, feats)
        contribs = (features_to_vector(feats) * dyn_weights).tolist()
        final.append({
            "rank": rank_pos + 1,
            "id": int(pid),
            "name": cand.get("name"),
            "age": cand.get("age"),
            "occupation": cand.get("occupation"),
            "height_cm": cand.get("height_cm"),
            "wealth_level": cand.get("wealth_level"),
            "well_settled": cand.get("well_settled"),
            "race_color": cand.get("race_color"),
            "emb_cosine": round(float(feats.get("emb_cosine", 0.0)), 4),
            "inter_overlap": int(feats.get("inter_overlap", 0)),
            "distance_km": round(float(feats.get("distance_km", 0.0)), 1),
            "height_diff": round(float(feats.get("height_diff", 0.0)), 1),
            "score": round(score, 4),
            "explain": labels,
            "icebreaker": ice,
            "contribs": contribs
        })
    return pd.DataFrame(final)

# -------------------------
# UI: Header, Lottie, CSS
# -------------------------
st.markdown("""
<style>
/* basic cards and badges */
.card { background: linear-gradient(180deg,#fff,#fbfdff); border-radius:12px; padding:14px; box-shadow:0 12px 30px rgba(16,24,40,0.06); border:1px solid rgba(99,102,241,0.06); margin-bottom:14px; }
.badge { display:inline-block; background:#eef2ff; color:#3730a3; padding:6px 10px; border-radius:999px; font-size:12px; margin-right:6px;}
.hero { display:flex; gap:18px; align-items:center; justify-content:space-between; padding:18px; border-radius:12px; background:linear-gradient(90deg, rgba(124,58,237,0.06), rgba(99,102,241,0.03)); margin-bottom:16px;}
.hero-title { font-size:22px; font-weight:700; color:#0f172a; margin:0;}
.small { font-size:13px; color:#6b7280;}
.pulse { animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{ transform:scale(1);} 50%{ transform:scale(1.02);} 100%{ transform:scale(1);} }
.progress-outer { background:#eef2ff; border-radius:999px; height:8px; overflow:hidden;}
.progress-inner { height:100%; background:linear-gradient(90deg,#7c3aed,#4f46e5); width:0%; transition: width 900ms cubic-bezier(.2,.9,.2,1);}
</style>
""", unsafe_allow_html=True)

# Lottie: small decorative animation (remote)
st.components.v1.html("""
<div class="hero">
  <div>
    <p class="hero-title">Matrimony MatchFinder — semantic + rules</p>
    <p class="small">Hybrid retrieval · Explainable signals · Preferences & requirements · Icebreakers</p>
  </div>
  <div>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_4v7o5b4v.json"  background="transparent"  speed="1"  style="width:180px; height:120px;"  loop  autoplay></lottie-player>
  </div>
</div>
""", height=160)

# -------------------------
# Sidebar form (user input + preferences)
# -------------------------
st.sidebar.header("Create your profile")
with st.sidebar.form("profile_form"):
    name = st.text_input("Full name", "")
    age = st.number_input("Age", min_value=18, max_value=80, value=28)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    lat = st.number_input("Latitude", value=17.3850, format="%.6f")
    lon = st.number_input("Longitude", value=78.4867, format="%.6f")
    bio = st.text_area("Short bio", "Family-oriented, career-focused, love traveling and music.")
    interests = st.text_input("Interests (comma separated)", "travel,music,reading")
    height_cm = st.number_input("Height (cm) (optional - leave 0 if unknown)", min_value=0, value=0)
    caste = st.selectbox("Caste (optional)", options=["", "CasteA", "CasteB", "CasteC"])
    race_color = st.selectbox("Race/Color (optional)", options=["", "Fair", "Wheatish", "Dark"])
    well_settled = st.checkbox("Are you well-settled?", value=False)

    st.markdown("---")
    st.markdown("## Requirements (hard constraints)")
    req_gender = st.selectbox("Looking for", options=["Opposite gender only", "Any gender (not typical)"])
    req_min_age = st.number_input("Preferred minimum age", min_value=18, max_value=80, value=18)
    req_max_age = st.number_input("Preferred maximum age", min_value=18, max_value=80, value=35)
    req_min_height = st.number_input("Minimum partner height (cm) (0 = no min)", min_value=0, value=0)
    req_same_caste = st.checkbox("Require same caste?", value=False)
    req_required_race = st.multiselect("Preferred race/color (optional)", options=["Fair", "Wheatish", "Dark"])
    req_well_settled = st.checkbox("Require partner to be well-settled?", value=False)

    st.markdown("---")
    st.markdown("## Soft preferences (affect scoring)")
    age_importance = st.slider("How important is age compatibility?", 0.0, 1.0, 0.3, 0.1)
    height_importance = st.slider("How important is height compatibility?", 0.0, 1.0, 0.2, 0.1)
    partner_age_pref = st.selectbox("Prefer partner older/younger?", options=["No preference", "Prefer older partner", "Prefer younger partner"])

    submit = st.form_submit_button("Find matches")

# -------------------------
# When user submits -> run matching pipeline
# -------------------------
if submit:
    # Build user profile dict
    user_profile = {
        "id": -1,
        "name": name or "You",
        "age": int(age) if age is not None else None,
        "gender": gender,
        "lat": float(lat),
        "lon": float(lon),
        "bio": bio,
        "interests": interests,
        "height_cm": int(height_cm) if height_cm and height_cm > 0 else None,
        "caste": caste or None,
        "race_color": race_color or None,
        "well_settled": bool(well_settled)
    }
    preferences = {
        "gender_pref": None,
        "max_distance_km": 200.0,
        "max_age_diff": max(1, req_max_age - req_min_age) if req_max_age >= req_min_age else 10,
        "caste_required": bool(req_same_caste),
        "min_height_cm": int(req_min_height) if req_min_height > 0 else None,
        "required_race": req_required_race if len(req_required_race) > 0 else None,
        "require_well_settled": bool(req_well_settled),
        "age_importance": float(age_importance),
        "height_importance": float(height_importance),
        "gender_age_preference": "partner_older" if partner_age_pref == "Prefer older partner" else ("partner_younger" if partner_age_pref == "Prefer younger partner" else None)
    }

    with st.spinner("Running hybrid retrieval, applying filters and ranking..."):
        results_df = run_match_for_user_profile(user_profile, preferences, candidate_pool=CANDIDATE_POOL, top_k=TOP_K, relax_if_empty=True)

    st.markdown(f"## Matches for **{user_profile['name']}**")
    if results_df.empty:
        st.warning("No matches found. Try relaxing strict requirements (caste / race / height) or increase max distance.")
    else:
        st.markdown(f"Showing top {len(results_df)} matches")
        # grid display: 2 columns
        cols = st.columns(2)
        for i, row in results_df.iterrows():
            col = cols[i % 2]
            with col:
                pulse = "pulse" if i == 0 else ""
                st.markdown(f"<div class='card {pulse}'>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                            f"<div><strong style='font-size:18px'>{row['name']}</strong> <div class='small'>{row['age']} yrs • {row['occupation'] or '—'}</div></div>"
                            f"<div style='text-align:right'><div style='color:#7c3aed;font-weight:800'>{row['score']}</div><div class='small'>Rank #{row['rank']}</div></div></div>",
                            unsafe_allow_html=True)
                # badges
                badges_html = " ".join([f"<span class='badge'>{b}</span>" for b in row['explain']])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown(f"<p class='small' style='margin-top:6px'>{row['icebreaker']}</p>", unsafe_allow_html=True)
                # emb strength bar
                emb_pct = int(max(0, min(100, float(row['emb_cosine']) * 100)))
                st.markdown("<div class='progress-outer' style='margin-top:8px;'><div class='progress-inner' style='width:{}%'></div></div>".format(emb_pct), unsafe_allow_html=True)
                # details expander
                with st.expander("View details & signals"):
                    st.write("Age:", row['age'])
                    st.write("Height (cm):", row.get("height_cm"))
                    st.write("Race/color:", row.get("race_color"))
                    st.write("Distance (km):", row['distance_km'])
                    st.write("Embedding cosine:", row.get("emb_cosine"))
                    # show top positive & negative contribs if present
                    contribs = row.get("contribs", None)
                    if contribs:
                        keys = ["emb_cosine","inter_overlap","same_religion","same_education","same_occupation_type",
                                "age_diff","distance_km","candidate_response_rate","profile_views_last_30d","age_compat_score",
                                "caste_match","race_match","settled_match","height_diff","height_compat_score"]
                        s = pd.Series(contribs, index=keys)
                        pos = s[s > 0].sort_values(ascending=False)[:6]
                        neg = s[s < 0].sort_values()[:4]
                        if not pos.empty:
                            st.markdown("**Top positive signals**")
                            for k, v in pos.items():
                                st.write(f"{k}: {v:.3f}")
                        if not neg.empty:
                            st.markdown("**Top negative signals**")
                            for k, v in neg.items():
                                st.write(f"{k}: {v:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Download results as CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button("Download CSV", csv, file_name="matches.csv", mime="text/csv")

st.markdown("---")
st.caption("Demo system — for production: replace synthetic DB, persist embeddings, add Cross-Encoder on GPU for best accuracy, and integrate real user behavior logs for training.")
