# outlook_dpr_run_and_monitor_latest_by_date.py
# Run: python outlook_dpr_run_and_monitor_latest_by_date.py

import os, time, pathlib, datetime as dt, re
import pandas as pd
import pythoncom
import win32com.client as win32

# ---------------- CONFIG ----------------
FOLDER_PATH = "Inbox/DPRs"           # <-- use exact Outlook path for your folder
DOWNLOAD_DIR = r"C:\Users\kaushikb\Documents\Work\Git\Office-work-\Raw Data\DPRs"
EMAIL_CONFIG_PATH = pathlib.Path(DOWNLOAD_DIR).parent / "Email_config.xlsx"

# Subject phrases (tolerant)
SUBJECT_PATTERNS = [
    r"\bdpr\b",
    r"\bdaily\W*progress\W*report\b",
    r"\bwork\W*progress\b",
]

# Attachment must contain "DPR" (tolerant)
ATTACHMENT_MUST_CONTAIN = [r"d\W*p\W*r"]   # dpr, d.p.r, d p r, d-p-r, etc.

# Allowed extensions (set to None to allow all)
ALLOWED_EXTS = {".xlsx",".xls"}

# Backfill window
BACKFILL_DAYS = 4
BACKFILL_ONLY_UNREAD = False
BACKFILL_MAX = 1000
# ---------------------------------------

def logprint(*args): print(*args, flush=True)
def norm(s): return (s or "").lower()

def _normalize_for_contains(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(value))

def _split_possible_subjects(value) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value)
    parts = re.split(r"[|,;\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def load_email_config(path: pathlib.Path = EMAIL_CONFIG_PATH) -> dict[str, dict]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Email configuration file not found at {path}")
    df = pd.read_excel(path)
    config: dict[str, dict] = {}
    for _, row in df.iterrows():
        email_raw = row.get("Email")
        if pd.isna(email_raw):
            continue
        email = str(email_raw).strip().lower()
        if not email:
            continue
        name_raw = row.get("Name")
        name = "" if pd.isna(name_raw) else str(name_raw).strip()
        config[email] = {
            "name": name,
            "possible_subjects": _split_possible_subjects(row.get("Possible_subject")),
        }
    return config

EMAIL_CONFIG = load_email_config()
ALLOWED_SENDERS = set(EMAIL_CONFIG.keys())

# --- Outlook helpers ---
def get_smtp_address(mail) -> str:
    try:
        addr = (mail.SenderEmailAddress or "").strip()
        if addr and "@" in addr: return addr.lower()
    except Exception: pass
    try:
        sender = getattr(mail, "Sender", None)
        if sender:
            prop = sender.PropertyAccessor
            smtp = prop.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x39FE001E")
            if smtp: return smtp.strip().lower()
    except Exception: pass
    return norm(getattr(mail, "SenderName", "") or "")

def is_mail_item(item) -> bool:
    try: return getattr(item, "Class", None) == 43  # olMailItem
    except Exception: return False

def subject_matches(subj: str, sender_email: str | None = None) -> bool:
    text = subj or ""
    lowered = norm(text)
    for pat in SUBJECT_PATTERNS:
        if re.search(pat, lowered, flags=re.IGNORECASE):
            return True
    if sender_email:
        sender_cfg = EMAIL_CONFIG.get(sender_email)
        if sender_cfg:
            normalized_subj = _normalize_for_contains(text)
            for candidate in sender_cfg.get("possible_subjects", []):
                candidate_norm = _normalize_for_contains(candidate)
                if candidate_norm and candidate_norm in normalized_subj:
                    return True
    return False

def should_process(mail) -> bool:
    if not is_mail_item(mail): return False
    sender = get_smtp_address(mail)
    if sender not in ALLOWED_SENDERS: return False
    subject = getattr(mail, "Subject", "") or ""
    if not subject_matches(subject, sender): return False
    return True

# --- Project code extraction ---
PROJECT_CODE_REGEXES = [
    r"\b(T[A-Z])\s*[-_. ]?\s*(\d{3,4})\b",   # TA 415, TB-416, TA415, TA.415, TC 1023
    r"\b(T[A-Z])[-_. ]?0?(\d{3})\b",          # TA_0415 / TA.0415 -> TA 415 (leading 0 tolerant for 3-4 digits)
]

def extract_project_code(text: str) -> str | None:
    t = text or ""
    for rgx in PROJECT_CODE_REGEXES:
        m = re.search(rgx, t, flags=re.IGNORECASE)
        if m:
            prefix = m.group(1).upper()
            num = int(m.group(2))  # normalizes 0415 -> 415
            return f"{prefix} {num}"
    return None

def derive_project_code(mail, attachment=None) -> str | None:
    """
    Try multiple sources (current attachment, subject, all attachment names)
    to determine the TA/TB style project identifier.
    """
    candidates: list[str] = []

    def push(value):
        if not value:
            return
        text = str(value).strip()
        if text:
            candidates.append(text)

    if attachment is not None:
        push(getattr(attachment, "FileName", None))
        push(getattr(attachment, "DisplayName", None))

    push(getattr(mail, "Subject", None))

    atts = getattr(mail, "Attachments", None)
    if atts:
        try:
            for i in range(1, atts.Count + 1):
                other = atts.Item(i)
                # Include other attachment names as fallbacks
                push(getattr(other, "FileName", None))
                push(getattr(other, "DisplayName", None))
        except Exception:
            pass

    for text in candidates:
        code = extract_project_code(text)
        if code:
            return code
    return None

# --- Date extraction ---
# Supports 2025-10-24, 24-10-2025, 24/10/2025, 24.10.2025, 20251024, 24 Oct 2025, Oct 24 2025, etc.
MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
MONTHS_FULL = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}

DATE_PATTERNS = [
    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    (re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b"), ("Y","M","D")),
    # DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b"), ("D","M","Y")),
    # YYYYMMDD (8 digits)
    (re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b"), ("Y","M","D")),
    # DD Mon YYYY
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})\s+(20\d{2})\b"), ("D","Mon3","Y")),
    # Mon DD YYYY
    (re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d{2})\b"), ("Mon","D","Y")),
]

def _to_int(s): 
    try: return int(s)
    except Exception: return None

def parse_date_from_text(text: str) -> dt.date | None:
    if not text: return None
    s = text
    for regex, order in DATE_PATTERNS:
        m = regex.search(s)
        if not m: 
            continue
        g = m.groups()
        Y=M=D=None
        for idx, key in enumerate(order):
            val = g[idx]
            if key == "Y": Y = _to_int(val)
            elif key == "M": M = _to_int(val)
            elif key == "D": D = _to_int(val)
            elif key == "Mon3": M = MONTHS.get(val[:3].lower())
            elif key == "Mon": 
                k = val.lower()
                M = MONTHS.get(k[:3]) or MONTHS_FULL.get(k)
        try:
            if Y and M and D:
                return dt.date(Y, M, D)
        except Exception:
            continue
    return None

def extract_report_date(mail, fallback_to_received=True) -> dt.date:
    # try attachment names first (likely to carry date), then subject
    try:
        atts = getattr(mail, "Attachments", None)
        if atts and atts.Count > 0:
            for i in range(1, atts.Count+1):
                nm = (atts.Item(i).FileName or "")
                d = parse_date_from_text(nm)
                if d: return d
    except Exception: pass
    d = parse_date_from_text(getattr(mail, "Subject", "") or "")
    if d: return d
    if fallback_to_received:
        try:
            # ReceivedTime is a COM datetime
            return getattr(mail, "ReceivedTime").date()
        except Exception:
            pass
    # As a final fallback, use today
    return dt.date.today()

# --- File naming, purge, save ---
def canonical_name(project_code: str, report_date: dt.date, ext: str) -> str:
    return f"{project_code} - DPR - {report_date:%Y-%m-%d}{ext}"

def _date_from_canonical_filename(filename: str, prefix: str, ext: str) -> dt.date | None:
    try:
        suffix_len = len(ext)
        date_slice = filename[len(prefix):-suffix_len] if suffix_len else filename[len(prefix):]
        return dt.datetime.strptime(date_slice, "%Y-%m-%d").date()
    except Exception:
        return None

def purge_previous_versions(download_dir: pathlib.Path, project_code: str, ext: str, report_date: dt.date) -> bool:
    """
    Remove older files for this project so only the newest remains.
    Returns False if an equal/newer file already exists and should be kept.
    """
    prefix = f"{project_code} - DPR - "
    prefix_lower = prefix.lower()
    for fn in os.listdir(download_dir):
        fl = fn.lower()
        if fl.startswith(prefix_lower) and fl.endswith(ext.lower()):
            existing_date = _date_from_canonical_filename(fn, prefix, ext)
            if existing_date and existing_date >= report_date:
                return False
            try: os.remove(download_dir / fn)
            except Exception: pass
    return True

def save_latest_for_mail(mail) -> list[str]:
    saved = []
    atts = getattr(mail, "Attachments", None)
    if not atts or atts.Count == 0: return saved

    # We’ll compute project per attachment (best accuracy), date once per mail (typical)
    mail_date = extract_report_date(mail, fallback_to_received=True)
    download_dir = pathlib.Path(DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, atts.Count + 1):
        att = atts.Item(i)
        name = att.FileName or ""
        if ALLOWED_EXTS and os.path.splitext(name)[1].lower() not in ALLOWED_EXTS:
            continue
        # must be a DPR file
        if not all(re.search(pat, norm(name), flags=re.IGNORECASE) for pat in ATTACHMENT_MUST_CONTAIN):
            continue

        # project code from attachment/subject/other attachments
        project = derive_project_code(mail, att)
        if not project:
            # If no project code, skip (or save with original name if you prefer)
            continue

        ext = os.path.splitext(name)[1].lower()
        # purge any previous versions for this project/ext; skip if a newer file already exists
        if not purge_previous_versions(download_dir, project, ext, mail_date):
            continue
        # save with canonical "<PROJECT> - DPR - <YYYY-MM-DD><ext>"
        target = download_dir / canonical_name(project, mail_date, ext)
        att.SaveAsFile(str(target))
        saved.append(str(target))
    return saved

# --- Folder resolving & plumbing ---
def get_folder_by_path(ns, path_str):
    p = path_str.strip().replace("\\", "/")
    while p.startswith("/"): p = p[1:]
    parts = [x for x in p.split("/") if x]
    if not parts: raise RuntimeError(f"Invalid path: '{path_str}'")
    if parts[0].lower() == "inbox":
        cur = ns.GetDefaultFolder(6)  # Inbox
        for name in parts[1:]: cur = cur.Folders[name]
        return cur
    first = parts[0].lower()
    store = None
    for s in ns.Folders:
        if s.Name.lower() == first or s.FolderPath.strip("\\").lower().endswith(first):
            store = s; break
    if store is None:
        for s in ns.Folders:
            if first in s.FolderPath.lower():
                store = s; break
    if store is None: raise RuntimeError(f"Mailbox/store not found for '{parts[0]}'")
    cur = store
    for name in parts[1:]: cur = cur.Folders[name]
    return cur

def backfill(folder):
    items = folder.Items
    items.Sort("[ReceivedTime]", True)
    since = dt.datetime.now() - dt.timedelta(days=BACKFILL_DAYS)
    r = items.Restrict(f"[ReceivedTime] >= '{since:%m/%d/%Y %I:%M %p}'")
    if BACKFILL_ONLY_UNREAD:
        r = r.Restrict("[Unread] = true")
    matched = saved = 0
    count = min(BACKFILL_MAX, r.Count)
    logprint(f"Backfill: last {BACKFILL_DAYS} day(s) in {folder.FolderPath} (up to {count} items)…")
    for i in range(1, count + 1):
        it = r.Item(i)
        if not should_process(it): continue
        files = save_latest_for_mail(it)
        if files:
            matched += 1; saved += len(files)
            logprint(f"  [BACKFILL] {get_smtp_address(it)} | {getattr(it,'Subject','')} -> {len(files)} file(s)")
    logprint(f"Backfill done. mails matched: {matched}, files saved: {saved}")

class ItemsEventHandler:
    def OnItemAdd(self, item):
        try:
            if should_process(item):
                files = save_latest_for_mail(item)
                if files:
                    logprint(f"[NEW] {get_smtp_address(item)} | {getattr(item,'Subject','')} -> {len(files)} file(s)")
        except Exception as e:
            logprint(f"ERROR in OnItemAdd: {e}")

def hook_folder_items(folder):
    items = folder.Items
    items.Sort("[ReceivedTime]", True)
    return win32.WithEvents(items, ItemsEventHandler)

def main():
    pathlib.Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    ns = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    folder = get_folder_by_path(ns, FOLDER_PATH)
    logprint(f"Target folder: {folder.FolderPath}")
    logprint(f"Saving to:     {DOWNLOAD_DIR}")
    backfill(folder)
    sink = hook_folder_items(folder)
    logprint("Monitoring for new mail… (Ctrl+C to stop)")
    try:
        while True:
            pythoncom.PumpWaitingMessages()
            time.sleep(0.4)
    except KeyboardInterrupt:
        logprint("Stopped by user.")

if __name__ == "__main__":
    main()
