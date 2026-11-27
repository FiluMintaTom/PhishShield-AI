
"""
Train PhishShield AI's URL model.

This creates a small transparent training dataset from labeled example URLs,
extracts the same features used at inference time, trains a RandomForest model,
prints a held-out evaluation, and saves the model + feature names.

IMPORTANT:
This starter dataset is for a project prototype/research baseline.
For a serious accuracy claim, replace/extend it with a larger, independently
labeled dataset and report precision, recall, F1 and ROC-AUC on an untouched
test set.
"""
import os, re, joblib
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE,"models")
os.makedirs(MODEL_DIR,exist_ok=True)

SUSPICIOUS={"login","verify","secure","account","bank","update","confirm",
            "password","wallet","payment","signin","bonus","free","support",
            "urgent","unlock","suspend","claim"}

def features(url):
    raw=url.strip()
    u=raw if re.match(r"^https?://",raw,re.I) else "https://"+raw
    p=urlparse(u)
    host=(p.hostname or "").lower()
    path=(p.path or "")
    query=(p.query or "")
    tokens=set(re.findall(r"[a-z0-9]+",u.lower()))
    suspicious=sum(1 for x in SUSPICIOUS if x in tokens)
    return {
      "url_length":len(u),
      "https":int(p.scheme=="https"),
      "suspicious_keywords":suspicious,
      "subdomain_count":max(0,len(host.split("."))-2),
      "hyphen_count":host.count("-"),
      "dot_count":host.count("."),
      "path_depth":len([x for x in path.split("/") if x]),
      "query_length":len(query),
      "has_ip":int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}",host))),
      "at_symbol":int("@" in u),
      "percent_encoded":u.count("%"),
      "digit_count":sum(c.isdigit() for c in u),
      "domain_length":len(host)
    }

# Starter labeled examples. These are deliberately simple research examples,
# not claims that every URL pattern here is always malicious/benign.
safe=[
"https://www.google.com","https://www.microsoft.com","https://www.apple.com",
"https://www.wikipedia.org","https://www.python.org","https://www.mozilla.org",
"https://www.github.com","https://www.cloudflare.com","https://www.openai.com",
"https://www.ubuntu.com","https://www.kernel.org","https://www.nasa.gov",
"https://www.bbc.com","https://www.ietf.org","https://www.w3.org",
"https://www.linkedin.com","https://www.adobe.com","https://www.ibm.com",
"https://www.cisco.com","https://www.oracle.com","https://www.nytimes.com",
"https://www.reddit.com","https://www.stackoverflow.com","https://www.docker.com"
]
phish=[
"http://secure-login-verify.com/account","http://verify-account-security.com/login",
"http://bank-login-confirm.com/update","http://paypal-secure-verify.com/login",
"http://account-password-reset.com/verify","http://free-bonus-claim.com/winner",
"http://urgent-bank-update.com/confirm","http://secure-wallet-login.com/signin",
"http://login-support-verify.com/account","http://payment-confirm-security.com/login",
"http://unlock-account-verify.com/password","http://signin-bank-update.com/secure",
"http://account-verify-now.com/login","http://claim-free-bonus.com/account",
"http://secure-login-account.com/verify","http://bank-secure-confirm.com/update",
"http://support-login-verify.com/account","http://password-reset-confirm.com/login",
"http://wallet-security-verify.com/signin","http://urgent-account-suspend.com/verify",
"http://secure-update-confirm.com/bank/login","http://login-verify-support.com/account",
"http://account-security-check.com/password","http://payment-update-verify.com/login",
"http://free-prize-claim.com/confirm"
]
rows=[(u,0) for u in safe]+[(u,1) for u in phish]
df=pd.DataFrame([features(u) for u,_ in rows]); y=pd.Series([label for _,label in rows])
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=.25,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,max_depth=8,class_weight="balanced",random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
print("Held-out accuracy:",round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred,target_names=["Safe","Phishing"],zero_division=0))
try: print("ROC-AUC:",round(roc_auc_score(y_test,prob),4))
except: pass
joblib.dump(model,os.path.join(MODEL_DIR,"url_model.joblib"))
joblib.dump(list(df.columns),os.path.join(MODEL_DIR,"feature_names.joblib"))
pd.DataFrame(rows,columns=["url","label"]).to_csv(os.path.join(BASE,"training_urls.csv"),index=False)
print("Saved models/url_model.joblib and models/feature_names.joblib")
