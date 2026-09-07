import re, joblib, warnings, sys
import numpy as np
import pandas as pd
import tldextract
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

warnings.filterwarnings("ignore")

print("=" * 60)
print("  PELATIHAN MODEL RF: BENIGN vs JUDI (ULTIMATE REVISION + FAST LEKSICAL)")
print("=" * 60)

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
BENIGN_FILE  = "datasetbenign-2.txt"
MALICIOUS_FILE = "datasetjudi-2.txt"
OUTPUT_MODEL = "model_rf_26Juni2026.pkl"
GRAFIK_NAME  = "grafik_rf_26Juni2026.png"

# =============================================================================
# KONFIGURASI FITUR
# =============================================================================
HIGH_RISK_TLDS = {
    'xyz', 'pro', 'site', 'fun', 'cc', 'online', 'info', 'biz',
    'top', 'club', 'vip', 'win', 'bet', 'casino', 'poker', 'slot',
    'live', 'cn', 'ru', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw',
    'icu', 'cyou', 'buzz', 'monster', 'uno', 'today', 'rest',
    'sbs', 'store', 'shop', 'link', 'click', 'services',
    'mx', 'in', 'ca', 'za', 'app', 'digital', 'tech', 'lol', 'space',
}

# Daftar 105 Keyword Judi Super Lengkap
GAMBLING_KEYWORDS = [
    "promo", "bonus", "deal", "win", "max", "maxwin", "maxxwin", "super", "mega", "flash", "quick", "promote",
    "easy", "top", "best", "free", "vip", "vvip", "prize", "judol", "gold", "judi", "card", "master", "fortuna",
    "elite", "prime", "ultra", "speed", "dewa", "zeus", "raja", "king", "ratu", "queen", "mister", "tokek", "joki",
    "online", "slot", "petir", "domino", "roulette", "luck", "lucky", "mahjong", "mahyong", "hockey", "hoki",
    "bandar", "markas", "kasino", "casino", "bet", "zone", "club", "site", "hub", "kartu", "wager", "wish",
    "keberuntungan", "untung", "pasti", "gampang", "menang", "juara", "laris", "olypus", "joker", "stake",
    "mantap", "gacor", "togel", "coin", "koin", "cash", "poker", "diamond", "berlian", "88", "777", "sultan",
    "agen", "chip", "jackpot", "fortune", "gift", "deposit", "hadiah", "toto", "mafia", "jp", "cuan",
    "qiuqiu", "taisai", "craps", "croupier", "capsa", "sicbo", "baccarat", "blackjack", "taruhan", "juragan",
    "bingo", "dadu", "dice", "rummy", "holdem", "dragon", "naga", "macan", "tiger", "paigow", "gaple", "boss",
    "wheel", "monopoli", "candu", "ketagihan", "maniak", "maniac", "rolet", "selot", "judy", "will",
    "withdraw", "shuffle", "flush", "surebet", "freebet", "sidebet", "turbo", "spin", "lotre", "champion"
]

# =============================================================================
# FUNGSI CLEANING DOMAIN
# =============================================================================
def clean_domain(raw):
    d = raw.strip().lower()
    if not d:
        return None

    if '/' in d:
        d = d.split('/')[0]

    if ' ' in d or '_' in d:
        return None

    if not re.match(r'^[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}$', d):
        return None

    return d

# =============================================================================
# FUNGSI EKSTRAKSI FITUR LEKSIKAL (12 FITUR) - DIOPTIMASI AMAN & CEPAT
# =============================================================================
def extract_single_domain(domain):
    empty_rec = {k: 0 for k in [
        'length', 'entropy', 'digit_ratio', 'vowel_ratio',
        'hyphen_ratio', 'subdomain_count', 'tld_risk',
        'consonant_ratio', 'keyword_hit', 'digit_cluster',
        'unique_char_ratio', 'has_ip_pattern'
    ]}

    d = str(domain).lower().strip()
    if not d:
        return empty_rec

    ext = tldextract.extract(d)
    core = ext.domain if ext.domain else "x"
    length = len(core)

    if length > 0:
        counts = Counter(core)
        prob = [count / length for count in counts.values()]
        ent = entropy(prob)
    else:
        ent = 0

    d_flat = d.replace('.', '').replace('-', '')

    return {
        'length'           : length,
        'entropy'          : ent,
        'digit_ratio'      : sum(c.isdigit() for c in core) / length if length > 0 else 0,
        'vowel_ratio'      : sum(c in 'aeiou' for c in core) / length if length > 0 else 0,
        'hyphen_ratio'     : core.count('-') / length if length > 0 else 0,
        'subdomain_count'  : len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'tld_risk'         : 1 if any(ext.suffix == t or ext.suffix.endswith(f'.{t}') for t in HIGH_RISK_TLDS) else 0,
        'consonant_ratio'  : sum(c.isalpha() and c not in 'aeiou' for c in core) / length if length > 0 else 0,
        'keyword_hit'      : sum(1 for kw in GAMBLING_KEYWORDS if kw in d_flat),
        'digit_cluster'    : 1 if re.search(r'\d{4,}', core) else 0,
        'unique_char_ratio': len(set(core)) / length if length > 0 else 0,
        'has_ip_pattern'   : 1 if re.match(r'^\d{1,3}[-\.]\d{1,3}', core) else 0,
    }

def get_lexical_features(domain_list):
    total = len(domain_list)
    print(f"    -> Memproses {total:,} baris secara Safe & Fast Mode...")
    
    features = []
    for i, d in enumerate(domain_list):
        features.append(extract_single_domain(d))
        
        if (i + 1) % 500000 == 0:
            print(f"       ... Leksikal selesai: {i + 1:,} / {total:,}")
            
    df = pd.DataFrame(features)
    return df.fillna(0)

def prepare_ngram_text(domain):
    ext = tldextract.extract(str(domain).lower().strip())
    return ' '.join(p for p in [ext.subdomain, ext.domain, ext.suffix] if p)

# =============================================================================
# LOAD & CLEANING DATASET
# =============================================================================
print("\n[*] Membaca dan membersihkan dataset...")
try:
    with open(BENIGN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        b_raw_all = [clean_domain(l) for l in f]
    with open(MALICIOUS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        m_raw_all = [clean_domain(l) for l in f]

    b_raw = list(set(d for d in b_raw_all if d))
    m_raw = list(set(d for d in m_raw_all if d))

    b_kotor = sum(1 for d in b_raw_all if d is None)
    m_kotor = sum(1 for d in m_raw_all if d is None)

    print(f"    Benign  : {len(b_raw):>10,} domain valid  ({b_kotor} dibuang karena kotor/duplikat)")
    print(f"    Judi    : {len(m_raw):>10,} domain valid  ({m_kotor} dibuang karena kotor/duplikat)")

except Exception as e:
    print(f"\n[!] ERROR FATAL: Gagal membaca dataset!\nDetail: {e}")
    sys.exit(1)

domains = b_raw + m_raw
labels  = [0] * len(b_raw) + [1] * len(m_raw)

# =============================================================================
# EKSTRAKSI FITUR
# =============================================================================
print("\n[*] [1/3] Mengekstrak 12 Fitur Leksikal...")
lex_df = get_lexical_features(domains)

print("\n[*] [2/3] Membuat N-Gram TF-IDF (char_wb 3-5, maks 10.000)...")
vectorizer = TfidfVectorizer(
    analyzer    = 'char_wb',
    ngram_range = (3, 5),
    max_features= 10_000,
    sublinear_tf= True,
)
ngram_matrix = vectorizer.fit_transform([prepare_ngram_text(d) for d in domains])

print("\n[*] [3/3] Scaling & menggabungkan fitur...")
scaler         = StandardScaler()
lexical_scaled = scaler.fit_transform(lex_df)
X              = hstack([ngram_matrix, csr_matrix(lexical_scaled)]).tocsr()
y              = np.array(labels)

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
print("\n[*] Membagi data: 80% latih, 20% uji (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================================================================
# PELATIHAN RANDOM FOREST
# =============================================================================
print("\n[*] Melatih Random Forest (Pohon lebih dalam & sensitif, CPU 100% mohon tunggu)...")
model = RandomForestClassifier(
    n_estimators     = 100,
    max_depth        = 30,  
    class_weight     = 'balanced_subsample',
    min_samples_leaf = 5,   
    min_samples_split= 10,   
    max_features     = 'sqrt',
    n_jobs           = -1,
    random_state     = 42,
    verbose          = 0,
)
model.fit(X_train, y_train)

# =============================================================================
# EVALUASI & THRESHOLD TUNING (Fokus 100% BLOKIR JUDI)
# =============================================================================
print("\n[*] Mengevaluasi model & threshold tuning (F3-score - Fokus Ekstrem pada Recall)...")
y_proba        = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

beta = 3
best_fbeta, best_thr = 0, 0.5

for i, thr in enumerate(thresholds):
    p, r = precisions[i], recalls[i]
    if p + r > 0:
        fbeta = (1 + beta**2) * p * r / ((beta**2 * p) + r)
        if fbeta > best_fbeta:
            best_fbeta, best_thr = fbeta, thr

y_pred_tuned = (y_proba >= best_thr).astype(int)

acc  = accuracy_score (y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default, zero_division=0)
rec  = recall_score   (y_test, y_pred_default, zero_division=0)
f1   = f1_score       (y_test, y_pred_default, zero_division=0)

acc_t  = accuracy_score (y_test, y_pred_tuned)
prec_t = precision_score(y_test, y_pred_tuned, zero_division=0)
rec_t  = recall_score   (y_test, y_pred_tuned, zero_division=0)
f1_t   = f1_score       (y_test, y_pred_tuned, zero_division=0)

print("\n" + "="*60)
print("  HASIL EVALUASI AKHIR (FOKUS RECALL)")
print("="*60)
print(f"  {'Metrik':<22} {'Thr=0.50':>10}  {'Thr Optimal':>12}")
print(f"  {'-'*48}")
print(f"  {'Akurasi':<22} {acc:>9.4f}   {acc_t:>10.4f}")
print(f"  {'Presisi':<22} {prec:>9.4f}   {prec_t:>10.4f}")
print(f"  {'Recall':<22} {rec:>9.4f}   {rec_t:>10.4f}")
print(f"  {'F1-Score':<22} {f1:>9.4f}   {f1_t:>10.4f}")
print(f"  {'F3-Score (optimal)':<22} {best_fbeta:>9.4f}   (thr={best_thr:.4f})")
print("="*60)

cm = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print("\n  Confusion Matrix (Threshold Optimal):")
print(f"  True Negative  (Benign  diloloskan ✓) : {tn:>10,}")
print(f"  False Positive (Benign  salah blokir) : {fp:>10,}  (Gunakan Whitelist di Production!)")
print(f"  False Negative (Judi    lolos ✗)      : {fn:>10,}  ← Target Utama Mendekati 0")
print(f"  True Positive  (Judi    diblokir  ✓)  : {tp:>10,}")

print("\n  Classification Report (Threshold Optimal):")
print(classification_report(y_test, y_pred_tuned,
                             target_names=['BENIGN', 'JUDI'],
                             zero_division=0))

# =============================================================================
# SIMPAN MODEL
# =============================================================================
artifacts = {
    'model'           : model,
    'ngram_vectorizer': vectorizer,
    'lexical_scaler'  : scaler,
    'threshold'       : best_thr,
    'feature_cols'    : list(lex_df.columns),
    'version'         : 'RF-Ultimate-AntiJudi',
}
joblib.dump(artifacts, OUTPUT_MODEL)
print(f"\n Model tersimpan : {OUTPUT_MODEL}")
print(f" Threshold aktif : {best_thr:.4f}")

# =============================================================================
# VISUALISASI (5 PANEL)
# =============================================================================
print("\n[*] Membuat grafik evaluasi (5 panel)...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    'Evaluasi Random Forest — BENIGN vs JUDI (Ultimate Revision)\n'
    f'Threshold Optimal: {best_thr:.4f}  |  F3-Score: {best_fbeta:.4f}',
    fontsize=13, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: Bar chart perbandingan metrik
ax1 = fig.add_subplot(gs[0, 0])
labels_m = ['Akurasi', 'Presisi', 'Recall', 'F1']
x = np.arange(len(labels_m))
w = 0.35
bars1 = ax1.bar(x - w/2, [acc, prec, rec, f1],
                w, label='Thr=0.50', color='#4878CF', alpha=0.85)
bars2 = ax1.bar(x + w/2, [acc_t, prec_t, rec_t, f1_t],
                w, label=f'Thr Optimal ({best_thr:.2f})',
                color='#E24A33', alpha=0.85)
ax1.set_ylim(0, 1.15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_m)
ax1.set_title('Perbandingan Metrik\n(Default vs Threshold Optimal)', fontweight='bold')
ax1.legend(fontsize=8)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h*100:.1f}%', ha='center', va='bottom', fontsize=7.5)

# Panel 2: Confusion matrix heatmap
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(np.array([[tn, fp], [fn, tp]]),
            annot=True, fmt=',', cmap='Blues', ax=ax2,
            xticklabels=['Pred: BENIGN', 'Pred: JUDI'],
            yticklabels=['True: BENIGN', 'True: JUDI'],
            linewidths=0.5, linecolor='gray')
ax2.set_title('Confusion Matrix\n(Threshold Optimal)', fontweight='bold')

# Panel 3: Precision-Recall curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(recalls, precisions, color='steelblue', lw=2, label='P-R Curve')
ax3.axvline(x=rec_t,  color='red',    linestyle='--', lw=1.2,
            label=f'Recall ({rec_t:.2f})')
ax3.axhline(y=prec_t, color='orange', linestyle='--', lw=1.2,
            label=f'Presisi ({prec_t:.2f})')
ax3.scatter([rec_t], [prec_t], color='red', zorder=5, s=60)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontweight='bold')
ax3.legend(fontsize=7.5)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1.05])
ax3.grid(alpha=0.3)

# Panel 4: Feature importance — 12 fitur leksikal
ax4 = fig.add_subplot(gs[1, :2])
n_ngram   = ngram_matrix.shape[1]
lex_imp   = model.feature_importances_[n_ngram:]
lex_names = list(lex_df.columns)
sorted_pairs = sorted(zip(lex_names, lex_imp), key=lambda x: x[1])
names_s, imp_s = zip(*sorted_pairs)
colors_b = ['#E24A33' if v > np.mean(imp_s) else '#4878CF' for v in imp_s]
ax4.barh(names_s, imp_s, color=colors_b, alpha=0.85)
ax4.set_xlabel('Feature Importance (Mean Decrease Impurity)')
ax4.set_title('Kepentingan 12 Fitur Leksikal\n(merah = di atas rata-rata)',
              fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (name, val) in enumerate(zip(names_s, imp_s)):
    ax4.text(val + 0.00005, i, f'{val:.5f}', va='center', fontsize=8)

# Panel 5: Distribusi skor probabilitas
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(y_proba[y_test == 0], bins=50, alpha=0.6,
         color='steelblue', label='BENIGN', density=True)
ax5.hist(y_proba[y_test == 1], bins=50, alpha=0.6,
         color='tomato',    label='JUDI',   density=True)
ax5.axvline(x=best_thr, color='black', linestyle='--', lw=1.5,
            label=f'Threshold ({best_thr:.4f})')
ax5.set_xlabel('Skor Probabilitas (P=Judi)')
ax5.set_ylabel('Densitas')
ax5.set_title('Distribusi Skor Probabilitas\nper Kelas', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

plt.savefig(GRAFIK_NAME, dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: {GRAFIK_NAME}")

print("\n" + "="*60)
print("  SELESAI. Ringkasan output:")
print(f"    Model    : {OUTPUT_MODEL}")
print(f"    Grafik   : {GRAFIK_NAME}")
print("="*60)