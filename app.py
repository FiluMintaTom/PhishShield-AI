
import os, re, sqlite3, secrets, hashlib, joblib, numpy as np, logging
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse
import requests
from flask import Flask, request, jsonify, session, render_template, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(APP_DIR, "phishshield.db")
MODEL_PATH=os.path.join(APP_DIR,"models","url_model.joblib")
FEATURE_PATH=os.path.join(APP_DIR,"models","feature_names.joblib")
URL_MODEL=None
URL_FEATURES=None
try:
    if os.path.exists(MODEL_PATH):
        URL_MODEL=joblib.load(MODEL_PATH)
        URL_FEATURES=joblib.load(FEATURE_PATH)
except Exception:
    logging.exception("Failed to load URL model or feature names")
    URL_MODEL=None

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("PHISHSHIELD_SECRET", secrets.token_hex(32))
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      full_name TEXT NOT NULL,
      username TEXT NOT NULL UNIQUE,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS scans(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      scan_type TEXT NOT NULL,
      content TEXT NOT NULL,
      prediction TEXT NOT NULL,
      risk INTEGER NOT NULL,
      confidence REAL NOT NULL,
      evidence_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS settings(
      user_id INTEGER PRIMARY KEY,
      theme TEXT NOT NULL DEFAULT 'dark',
      language TEXT NOT NULL DEFAULT 'English',
      alerts INTEGER NOT NULL DEFAULT 1,
      completion INTEGER NOT NULL DEFAULT 1,
      reports INTEGER NOT NULL DEFAULT 0,
      system INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    c.commit(); c.close()

def current_user():
    uid = session.get("user_id")
    if not uid: return None
    c=db(); u=c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone(); c.close()
    return u

def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not current_user(): return jsonify({"error":"Authentication required"}), 401
        return fn(*a, **kw)
    return wrapper

def now():
    return datetime.now(timezone.utc).isoformat()

def valid_email(x):
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", x or ""))

def valid_password(x):
    return bool(x and len(x)>=8 and re.search(r"[A-Za-z]",x) and re.search(r"\d",x))

def domain_from_url(u):
    try:
        p=urlparse(u if re.match(r"^https?://",u,re.I) else "https://"+u)
        return p.hostname.lower() if p.hostname else ""
    except: return ""

def rdap_domain(domain):
    try:
        r=requests.get(f"https://rdap.org/domain/{domain}", timeout=7,
                       headers={"User-Agent":"PhishShieldAI/1.0"})
        if r.ok:
            data=r.json()
            events=data.get("events",[])
            created=next((e["eventDate"] for e in events if e.get("eventAction")=="registration"),None)
            return {"available":True,"created":created,"source":"RDAP","raw_status":r.status_code}
        return {"available":False,"reason":f"RDAP HTTP {r.status_code}"}
    except Exception as e:
        return {"available":False,"reason":"RDAP lookup unavailable"}


def model_feature_dict(e):
    # Keep this extraction identical to train_model.py:features().
    u=e["normalized_url"]
    p=urlparse(u)
    host=e["domain"]
    tokens=set(re.findall(r"[a-z0-9]+",u.lower()))
    suspicious_terms={"login","verify","secure","account","bank","update","confirm",
                      "password","wallet","payment","signin","bonus","free","support",
                      "urgent","unlock","suspend","claim"}
    return {
      "url_length":len(u), "https":int(p.scheme=="https"),
      "suspicious_keywords":sum(1 for word in suspicious_terms if word in tokens),
      "subdomain_count":max(0,len(host.split("."))-2), "hyphen_count":host.count("-"),
      "dot_count":host.count("."), "path_depth":len([x for x in (p.path or "").split("/") if x]),
      "query_length":len(p.query or ""),
      "has_ip":int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}",host))),
      "at_symbol":int("@" in u), "percent_encoded":u.count("%"),
      "digit_count":sum(c.isdigit() for c in u), "domain_length":len(host)
    }

def scan_url(raw):
    u=raw.strip()
    if not re.match(r"^https?://",u,re.I): u="https://"+u
    p=urlparse(u)
    if p.scheme not in ("http","https") or not p.hostname:
        raise ValueError("Enter a valid HTTP/HTTPS URL.")
    domain=p.hostname.lower()
    suspicious_words=sorted(set(re.findall(
        r"(login|verify|secure|account|bank|update|confirm|password|wallet|payment|signin|bonus|free|support|urgent)",
        u, re.I)))
    path_complexity = len(p.path.split("/")) + len(p.query)
    url_length=len(u)
    hyphens=domain.count("-")
    subdomains=max(0, len(domain.split("."))-2)
    https=p.scheme=="https"

    redirect_count=None
    final_url=None
    status=None
    http_error=None
    try:
        r=requests.get(u, timeout=8, allow_redirects=True, stream=True,
                       headers={"User-Agent":"PhishShieldAI/1.0"})
        redirect_count=len(r.history)
        final_url=r.url
        status=r.status_code
        r.close()
    except Exception:
        http_error="Target could not be reached from the server."

    rd=rdap_domain(domain)
    domain_age_days=None
    if rd.get("created"):
        try:
            created=datetime.fromisoformat(rd["created"].replace("Z","+00:00"))
            domain_age_days=max(0,(datetime.now(timezone.utc)-created).days)
        except: pass

    reasons=[]
    mf=model_feature_dict({
        "url_length":url_length,"https":https,"suspicious_keywords":suspicious_words,
        "domain":domain,"url_complexity":path_complexity,"normalized_url":u
    })
    if URL_MODEL is not None:
        X=np.array([[mf.get(n,0) for n in URL_FEATURES]],dtype=float)
        probability=float(URL_MODEL.predict_proba(X)[0][1])
        risk=round(probability*100)
        pred="Phishing" if probability>=0.5 else "Safe"
        confidence=round(max(probability,1-probability)*100,1)
        model_method="RandomForest URL model trained from the included starter labeled dataset."
        from shap_explainer import explain
        try:
            shap_values=explain(mf)
        except Exception:
            logging.exception("SHAP explanation failed for scanned URL %r", u)
            shap_values=[]
    else:
        # Safe fallback if the model has not been trained yet.
        risk=0
        if suspicious_words: risk += min(35,8*len(suspicious_words)); reasons.append("Suspicious security/account language was found in the URL.")
        if not https: risk += 15; reasons.append("The URL does not use HTTPS.")
        if url_length>100: risk += 10; reasons.append("The URL is unusually long.")
        if subdomains>=3: risk += 10; reasons.append("The hostname contains several subdomain levels.")
        if domain_age_days is not None and domain_age_days<30: risk += 25; reasons.append(f"The domain registration age is about {domain_age_days} days.")
        if redirect_count is not None and redirect_count>=3: risk += 10; reasons.append(f"The request followed {redirect_count} redirects.")
        risk=min(100,risk); pred="Phishing" if risk>=50 else "Safe"
        confidence=round(max(51,70+abs(risk-50)*.55),1); model_method="Transparent feature-based fallback; train the included model for ML mode."; shap_values=[]

    evidence={
      "normalized_url":u, "domain":domain, "https":https,
      "url_length":url_length, "suspicious_keywords":suspicious_words,
      "url_complexity":path_complexity, "redirect_count":redirect_count,
      "final_url":final_url, "http_status":status, "http_error":http_error,
      "domain_age_days":domain_age_days, "rdap":rd,
      "reasons":reasons,
      "method":model_method,"shap_values":shap_values
    }
    return pred,risk,confidence,evidence

def email_scan(subject,sender,body):
    text=" ".join([subject or "", sender or "", body or ""])
    patterns={
      "Urgency language":r"\b(urgent|immediately|act now|within \d+ hours|suspend)\b",
      "Credential request":r"\b(password|otp|one[- ]time|verify your account|login)\b",
      "Payment request":r"\b(payment|invoice|bank|card|wallet|transfer)\b",
      "Prize/social engineering":r"\b(prize|winner|gift|bonus|claim|click here)\b",
      "Link instruction":r"\b(click|open|follow)\b"
    }
    hits=[name for name,pat in patterns.items() if re.search(pat,text,re.I)]
    sender_domain=sender.split("@")[-1].lower() if "@" in sender else None
    risk=min(100, len(hits)*20 + (10 if sender and not valid_email(sender) else 0))
    pred="Phishing" if risk>=50 else "Safe"
    confidence=round(min(98, max(55, 70+abs(risk-50)*0.5)),1)
    evidence={"indicators":hits,"sender_domain":sender_domain,
              "reasons":[f"{x} detected." for x in hits],
              "method":"Transparent text-pattern analysis; not a claim of laboratory-validated ML accuracy."}
    return pred,risk,confidence,evidence

def save_scan(user, typ, content, pred, risk, conf, evidence):
    c=db()
    c.execute("""INSERT INTO scans(user_id,scan_type,content,prediction,risk,confidence,evidence_json,created_at)
                 VALUES(?,?,?,?,?,?,?,?)""",
              (user["id"],typ,content,pred,risk,conf,__import__("json").dumps(evidence),now()))
    c.commit(); c.close()

@app.route("/")
def index(): return render_template("index.html")

@app.post("/api/register")
def register():
    d=request.get_json() or {}
    name=d.get("fullName","").strip(); username=d.get("username","").strip()
    email=d.get("email","").strip().lower(); pw=d.get("password",""); cp=d.get("confirmPassword","")
    if len(name)<2 or not re.fullmatch(r"[A-Za-z .'-]{2,80}",name): return jsonify({"error":"Enter a valid full name."}),400
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}",username): return jsonify({"error":"Username must be 3-30 characters and use letters, numbers, _, . or -."}),400
    if not valid_email(email): return jsonify({"error":"Enter a valid email address."}),400
    if not valid_password(pw): return jsonify({"error":"Password must be at least 8 characters and contain letters and a number."}),400
    if pw!=cp: return jsonify({"error":"Passwords do not match."}),400
    c=db()
    try:
        cur=c.execute("INSERT INTO users(full_name,username,email,password_hash,created_at) VALUES(?,?,?,?,?)",
                      (name,username,email,generate_password_hash(pw),now()))
        uid=cur.lastrowid
        c.execute("INSERT INTO settings(user_id) VALUES(?)",(uid,))
        c.commit()
    except sqlite3.IntegrityError:
        c.close(); return jsonify({"error":"Username or email already exists."}),409
    c.close(); session["user_id"]=uid
    return jsonify({"ok":True})

@app.post("/api/login")
def login():
    d=request.get_json() or {}; identity=d.get("identity","").strip().lower(); pw=d.get("password","")
    c=db(); u=c.execute("SELECT * FROM users WHERE lower(email)=? OR lower(username)=?",(identity,identity)).fetchone(); c.close()
    if not u or not check_password_hash(u["password_hash"],pw): return jsonify({"error":"Invalid username/email or password."}),401
    session["user_id"]=u["id"]; return jsonify({"ok":True})

@app.post("/api/logout")
def api_logout(): session.clear(); return jsonify({"ok":True})

@app.get("/api/me")
def me():
    u=current_user()
    if not u:return jsonify({"authenticated":False})
    return jsonify({"authenticated":True,"fullName":u["full_name"],"username":u["username"],"email":u["email"]})

@app.post("/api/scan/url")
@login_required
def api_url():
    try:
        u=(request.get_json() or {}).get("url","")
        pred,risk,conf,evidence=scan_url(u)
        user=current_user(); save_scan(user,"Website",u,pred,risk,conf,evidence)
        return jsonify({"prediction":pred,"risk":risk,"confidence":conf,"evidence":evidence})
    except ValueError as e:return jsonify({"error":str(e)}),400
    except Exception as e:
        app.logger.exception("URL scan failed")
        return jsonify({"error":"Scan failed safely. Please try again."}),500

@app.post("/api/scan/email")
@login_required
def api_email():
    d=request.get_json() or {}; subject=d.get("subject","").strip(); sender=d.get("sender","").strip(); body=d.get("body","").strip()
    if len(subject)>300 or len(sender)>320 or len(body)>100000:return jsonify({"error":"Input is too large."}),400
    if not body and not subject:return jsonify({"error":"Enter email content."}),400
    pred,risk,conf,evidence=email_scan(subject,sender,body)
    save_scan(current_user(),"Email",subject or body[:120],pred,risk,conf,evidence)
    return jsonify({"prediction":pred,"risk":risk,"confidence":conf,"evidence":evidence})

@app.get("/api/scans")
@login_required
def scans():
    c=db(); rows=c.execute("SELECT id,scan_type,content,prediction,risk,confidence,created_at,evidence_json FROM scans WHERE user_id=? ORDER BY id DESC",(current_user()["id"],)).fetchall(); c.close()
    return jsonify([dict(r) for r in rows])

@app.get("/api/dashboard")
@login_required
def dashboard():
    c=db(); uid=current_user()["id"]
    rows=c.execute("SELECT prediction,created_at FROM scans WHERE user_id=?",(uid,)).fetchall()
    c.close()
    total=len(rows); ph=sum(r["prediction"]=="Phishing" for r in rows); safe=total-ph
    days={}
    for r in rows:
        day=r["created_at"][:10]; days[day]=days.get(day,0)+1
    return jsonify({"total":total,"phishing":ph,"safe":safe,"daily":days})

@app.get("/api/settings")
@login_required
def get_settings():
    c=db(); r=c.execute("SELECT * FROM settings WHERE user_id=?",(current_user()["id"],)).fetchone(); c.close()
    return jsonify(dict(r))

@app.post("/api/settings")
@login_required
def set_settings():
    d=request.get_json() or {}; allowed_theme={"dark","light","system"}; allowed_lang={"English","Malayalam"}
    theme=d.get("theme","dark"); lang=d.get("language","English")
    if theme not in allowed_theme or lang not in allowed_lang:return jsonify({"error":"Invalid setting."}),400
    c=db(); c.execute("""UPDATE settings SET theme=?,language=?,alerts=?,completion=?,reports=?,system=? WHERE user_id=?""",
      (theme,lang,int(bool(d.get("alerts"))),int(bool(d.get("completion"))),int(bool(d.get("reports"))),int(bool(d.get("system"))),current_user()["id"]))
    c.commit(); c.close(); return jsonify({"ok":True})

if __name__=="__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
