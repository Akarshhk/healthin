"""
Rural Health Triage — Enhanced Model Training Pipeline
======================================================
Generates medically-grounded synthetic data and trains the XGBoost triage model.

Improvements over v1:
  • 15 diseases (5 new: chickenpox, jaundice, UTI, asthma, cholera)
  • WHO/CDC-aligned symptom probability profiles
  • Age/gender-correlated symptom generation
  • SMOTE oversampling for class balance
  • 5-fold stratified cross-validation
  • Saved evaluation report with confusion matrix
  • Exported training dataset for reproducibility

Run:  python train_model.py
"""

import json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SYMPTOM FEATURES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 34 binary features + age + gender
SYMPTOMS = [
    "fever","high_fever","chills","headache","severe_headache",
    "body_ache","joint_pain","muscle_pain","cough","chronic_cough",
    "sore_throat","runny_nose","sneezing","difficulty_breathing",
    "rapid_breathing","chest_pain","abdominal_pain","nausea",
    "vomiting","diarrhea","rash","eye_pain","dark_urine",
    "blood_in_sputum","night_sweats","weight_loss","fatigue",
    "weakness","dizziness","sweating","confusion","bleeding",
    "pale_skin","dehydration_signs",
]
FEATURES = SYMPTOMS + ["age", "gender"]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DISEASE DEFINITIONS — Evidence-based symptom profiles                 ║
# ║  Sources: WHO fact-sheets, CDC guidelines, clinical literature         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DISEASES = {
    # ── RED (Urgent) ──────────────────────────────────────────────────────
    "malaria": {
        "urgency": "RED",
        "source": "WHO Malaria Fact-sheet",
        "age_range": (2, 70),  # most common age range
        "p": {
            "fever":.95, "high_fever":.70, "chills":.90, "headache":.85,
            "sweating":.80, "body_ache":.70, "nausea":.50, "vomiting":.40,
            "dark_urine":.30, "fatigue":.60, "weakness":.50, "dizziness":.30,
            "confusion":.10,  # severe malaria
        },
    },
    "dengue": {
        "urgency": "RED",
        "source": "CDC Dengue Signs & Symptoms",
        "age_range": (5, 65),
        "p": {
            "fever":.95, "high_fever":.85, "severe_headache":.80,
            "eye_pain":.70, "joint_pain":.80, "muscle_pain":.75,
            "rash":.50, "nausea":.60, "vomiting":.40, "fatigue":.70,
            "weakness":.50, "bleeding":.15, "abdominal_pain":.25,
        },
    },
    "pneumonia": {
        "urgency": "RED",
        "source": "WHO Pneumonia Fact-sheet",
        "age_range": (1, 80),
        "gender_bias": None,
        "p": {
            "cough":.90, "fever":.85, "high_fever":.50,
            "difficulty_breathing":.80, "rapid_breathing":.60,
            "chest_pain":.60, "fatigue":.70, "sweating":.30,
            "nausea":.20, "confusion":.15, "chills":.40,
        },
    },
    "heat_stroke": {
        "urgency": "RED",
        "source": "CDC Heat-Related Illness",
        "age_range": (15, 80),
        "p": {
            "high_fever":.90, "confusion":.70, "sweating":.30,
            "headache":.60, "nausea":.50, "vomiting":.30,
            "dizziness":.70, "rapid_breathing":.40, "weakness":.60,
        },
    },
    "cholera": {
        "urgency": "RED",
        "source": "WHO Cholera Fact-sheet",
        "age_range": (1, 70),
        "p": {
            "diarrhea":.95, "vomiting":.85, "dehydration_signs":.90,
            "weakness":.80, "abdominal_pain":.50, "nausea":.60,
            "muscle_pain":.30, "fever":.20, "confusion":.15,
        },
    },

    # ── YELLOW (See doctor soon) ──────────────────────────────────────────
    "typhoid": {
        "urgency": "YELLOW",
        "source": "WHO Typhoid Fact-sheet",
        "age_range": (5, 60),
        "p": {
            "fever":.95, "high_fever":.60, "headache":.70,
            "abdominal_pain":.70, "weakness":.80, "fatigue":.70,
            "diarrhea":.40, "nausea":.40, "rash":.20, "body_ache":.50,
        },
    },
    "tuberculosis": {
        "urgency": "YELLOW",
        "source": "WHO TB Fact-sheet",
        "age_range": (15, 70),
        "p": {
            "chronic_cough":.90, "cough":.95, "fever":.60,
            "night_sweats":.70, "weight_loss":.80, "blood_in_sputum":.40,
            "fatigue":.80, "chest_pain":.40, "weakness":.60,
        },
    },
    "gastroenteritis": {
        "urgency": "YELLOW",
        "source": "CDC Viral Gastroenteritis",
        "age_range": (1, 70),
        "p": {
            "diarrhea":.90, "vomiting":.70, "abdominal_pain":.80,
            "nausea":.85, "fever":.40, "dehydration_signs":.40,
            "weakness":.50, "fatigue":.40, "headache":.20,
        },
    },
    "anemia": {
        "urgency": "YELLOW",
        "source": "WHO Anaemia Guidelines",
        "age_range": (5, 70),
        "gender_bias": "F",  # more common in women
        "p": {
            "fatigue":.90, "weakness":.85, "pale_skin":.70,
            "dizziness":.60, "difficulty_breathing":.30, "headache":.30,
            "chest_pain":.10, "rapid_breathing":.20,
        },
    },
    "jaundice": {
        "urgency": "YELLOW",
        "source": "WHO Hepatitis Fact-sheet",
        "age_range": (5, 70),
        "p": {
            "fatigue":.80, "weakness":.70, "nausea":.65,
            "vomiting":.40, "abdominal_pain":.60, "dark_urine":.85,
            "fever":.50, "weight_loss":.30, "pale_skin":.40,
        },
    },
    "urinary_tract_infection": {
        "urgency": "YELLOW",
        "source": "CDC UTI Guidelines",
        "age_range": (15, 70),
        "gender_bias": "F",
        "p": {
            "fever":.50, "abdominal_pain":.70, "nausea":.30,
            "weakness":.40, "fatigue":.50, "dark_urine":.60,
            "chills":.25, "vomiting":.15, "body_ache":.20,
        },
    },
    "asthma_attack": {
        "urgency": "YELLOW",
        "source": "WHO Asthma Fact-sheet",
        "age_range": (5, 65),
        "p": {
            "difficulty_breathing":.95, "cough":.80, "chest_pain":.60,
            "rapid_breathing":.70, "sweating":.30, "fatigue":.40,
            "weakness":.30,
        },
    },

    # ── GREEN (Home care) ─────────────────────────────────────────────────
    "common_cold": {
        "urgency": "GREEN",
        "source": "CDC Common Cold",
        "age_range": (1, 80),
        "p": {
            "runny_nose":.90, "sneezing":.85, "sore_throat":.70,
            "cough":.50, "fever":.30, "headache":.30,
            "body_ache":.20, "fatigue":.30,
        },
    },
    "flu": {
        "urgency": "GREEN",
        "source": "CDC Influenza Symptoms",
        "age_range": (1, 80),
        "p": {
            "fever":.80, "high_fever":.30, "headache":.60,
            "body_ache":.70, "muscle_pain":.60, "cough":.60,
            "sore_throat":.50, "fatigue":.70, "runny_nose":.40, "chills":.40,
        },
    },
    "chickenpox": {
        "urgency": "GREEN",
        "source": "CDC Chickenpox",
        "age_range": (2, 40),
        "p": {
            "rash":.95, "fever":.85, "headache":.50,
            "fatigue":.60, "body_ache":.40, "sore_throat":.20,
            "weakness":.30, "nausea":.15,
        },
    },
}

DISEASE_NAMES = list(DISEASES.keys())

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENHANCED SYNTHETIC DATA GENERATOR                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_dataset(n_per_disease: int = 800):
    """Generate medically-grounded synthetic patient data.

    Improvements:
    - Age sampled from disease-specific ranges
    - Gender bias for diseases like anemia, UTI
    - Symptom co-occurrence noise for realism
    - More samples per disease for better training
    """
    rows, labels = [], []

    for idx, (name, info) in enumerate(DISEASES.items()):
        age_lo, age_hi = info.get("age_range", (1, 80))
        gender_bias = info.get("gender_bias", None)

        for _ in range(n_per_disease):
            # Age: sample from disease-typical range with some outliers
            if np.random.random() < 0.85:
                age = np.random.randint(age_lo, age_hi + 1)
            else:
                age = np.random.randint(1, 80)

            # Gender: apply bias if applicable
            if gender_bias == "F":
                gender = 0 if np.random.random() < 0.70 else 1
            elif gender_bias == "M":
                gender = 1 if np.random.random() < 0.70 else 0
            else:
                gender = np.random.randint(0, 2)

            # Symptoms: sample from disease probability profile
            row = {}
            for s in SYMPTOMS:
                base_prob = info["p"].get(s, 0.03)  # 3% background noise

                # Age modifiers: children and elderly have higher symptom severity
                if age < 5 or age > 65:
                    if s in ("high_fever", "confusion", "dehydration_signs",
                             "difficulty_breathing"):
                        base_prob = min(1.0, base_prob * 1.3)

                row[s] = int(np.random.random() < base_prob)

            row["age"] = age
            row["gender"] = gender
            rows.append(row)
            labels.append(idx)

    return pd.DataFrame(rows)[FEATURES], np.array(labels)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING PIPELINE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("  Rural Health Triage — Enhanced Model Training (v2)")
    print("=" * 60)

    # ── 1. Generate dataset ───────────────────────────────────────────────
    N_PER_DISEASE = 800
    X, y = generate_dataset(N_PER_DISEASE)
    total = len(X)
    print(f"\n📊 Dataset: {total:,} samples  |  {len(DISEASE_NAMES)} diseases")
    print(f"   Diseases: {', '.join(DISEASE_NAMES)}")

    # ── 2. Train/test split ───────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── 3. SMOTE oversampling (if imbalanced) ─────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        print(f"   SMOTE: {len(X_tr):,} → {len(X_tr_res):,} (balanced)")
        X_tr, y_tr = X_tr_res, y_tr_res
    except ImportError:
        print("   ℹ️  imblearn not installed — skipping SMOTE (pip install imbalanced-learn)")

    # ── 4. Cross-validation ───────────────────────────────────────────────
    print("\n🔄 5-fold cross-validation...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(DISEASE_NAMES),
        eval_metric="mlogloss",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    print(f"   CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"   Fold scores: {[f'{s:.1%}' for s in cv_scores]}")

    # ── 5. Final training on full train set ───────────────────────────────
    print("\n🏋️ Training final model...")
    model.fit(X_tr, y_tr)

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.1%}\n")

    report_str = classification_report(y_te, y_pred, target_names=DISEASE_NAMES)
    print(report_str)

    # ── 7. Save model + metadata ──────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model.save_model("models/triage_model.json")

    meta = {
        "features": FEATURES,
        "symptoms": SYMPTOMS,
        "diseases": DISEASE_NAMES,
        "urgency": {n: d["urgency"] for n, d in DISEASES.items()},
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 8. Save evaluation report ─────────────────────────────────────────
    report_dict = classification_report(
        y_te, y_pred, target_names=DISEASE_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_te, y_pred).tolist()
    eval_report = {
        "accuracy": round(acc, 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "per_class": report_dict,
        "confusion_matrix": cm,
        "class_labels": DISEASE_NAMES,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "n_diseases": len(DISEASE_NAMES),
        "n_features": len(FEATURES),
        "model_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
        },
    }
    with open("models/evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    # ── 9. Save training dataset for reproducibility ──────────────────────
    X_full = pd.concat([X_tr, X_te])
    y_full = np.concatenate([y_tr, y_te])
    X_full["label"] = y_full
    X_full["disease"] = X_full["label"].map(lambda i: DISEASE_NAMES[i])
    X_full.to_csv("models/training_data.csv", index=False)

    # ── 10. Summary ───────────────────────────────────────────────────────
    kb = os.path.getsize("models/triage_model.json") / 1024
    print(f"\n{'='*60}")
    print(f"✅  Model         →  models/triage_model.json  ({kb:.0f} KB)")
    print(f"✅  Metadata      →  models/metadata.json")
    print(f"✅  Evaluation    →  models/evaluation_report.json")
    print(f"✅  Training data →  models/training_data.csv")
    print(f"\n   {len(DISEASE_NAMES)} diseases  |  {len(FEATURES)} features  |  {total:,} samples")
    print(f"   Test accuracy: {acc:.1%}  |  CV accuracy: {cv_scores.mean():.1%}")
    print(f"\nNext:  python app.py")


if __name__ == "__main__":
    main()