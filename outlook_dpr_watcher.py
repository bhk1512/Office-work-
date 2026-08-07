# outlook_dpr_run_and_monitor_latest_by_date.py
# Run: python outlook_dpr_run_and_monitor_latest_by_date.py

import os, time, pathlib, datetime as dt, re, argparse
import pandas as pd
import pythoncom
import win32com.client as win32

from dashboard.project_identity import (
    canonical_dpr_filename,
    normalize_line_name,
    parse_project_identity_from_filename,
)
from dashboard.completed_projects import (
    is_completed_project,
    load_completed_project_keys,
)

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
ALLOWED_EXTS = {".xlsx", ".xls", ".xlsm"}

# Backfill window
BACKFILL_DAYS = 4
BACKFILL_ONLY_UNREAD = False
BACKFILL_MAX = 1000
# ---------------------------------------

def logprint(*args): print(*args, flush=True)
def norm(s): return (s or "").lower()

def _normalize_for_contains(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(value))

def _normalize_attachment_key(value: str, *, drop_digits: bool = False) -> str:
    """
    Normalize attachment text for substring checks by stripping extension, special chars,
    and optionally digits (dates in filenames).
    """
    text = str(value or "")
    text = os.path.splitext(text)[0]
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    if drop_digits:
        text = re.sub(r"\d+", "", text)
    return text

_ATTACHMENT_TOKEN_STOPWORDS = {"and", "of", "the", "for", "to", "in", "on", "at", "with"}


def _tokenize_attachment_key(value: str, *, drop_digits: bool = False) -> list[str]:
    """
    Tokenize attachment text for resilient contains checks where extra words may
    exist between expected phrases.
    """
    text = str(value or "")
    text = os.path.splitext(text)[0]
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens: list[str] = []
    for token in raw_tokens:
        if drop_digits:
            token = re.sub(r"\d+", "", token)
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


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

def _split_semicolon_values(value) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    parts = [part.strip() for part in str(value).split(";")]
    return [part for part in parts if part]


def _build_attachment_rules(row) -> list[dict]:
    attachment_names = _split_semicolon_values(row.get("Possible_Attachement_Name"))
    if not attachment_names:
        return []

    line_names = _split_semicolon_values(row.get("Line_Name"))
    if line_names and len(line_names) != len(attachment_names):
        logprint(
            "[email-config] warning: skipping line-aware attachment mapping due to count mismatch "
            f"(attachments={len(attachment_names)}, lines={len(line_names)}) for email={row.get('Email', '')}"
        )
        line_names = []

    rules: list[dict] = []
    for idx, attachment_name in enumerate(attachment_names):
        line_name = line_names[idx] if idx < len(line_names) else ""
        rules.append(
            {
                "match_text": attachment_name,
                "match_key": _normalize_attachment_key(attachment_name, drop_digits=True),
                "match_tokens": _tokenize_attachment_key(attachment_name, drop_digits=True),
                "line_name": normalize_line_name(line_name),
                "line_key": _normalize_for_contains(line_name),
                "row_index": idx,
            }
        )
    return rules


def load_email_config(path: pathlib.Path = EMAIL_CONFIG_PATH) -> dict[str, list[dict]]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Email configuration file not found at {path}")
    df = pd.read_excel(path)
    config: dict[str, list[dict]] = {}
    for row_idx, row in df.iterrows():
        email_raw = row.get("Email")
        if pd.isna(email_raw):
            continue
        email = str(email_raw).strip().lower()
        if not email:
            continue
        name_raw = row.get("Name")
        name = "" if pd.isna(name_raw) else str(name_raw).strip()
        project_code_raw = row.get("Project_Code")
        project_code = "" if pd.isna(project_code_raw) else str(project_code_raw).strip().upper()
        entry = {
            "name": name,
            "possible_subjects": _split_possible_subjects(row.get("Possible_subject")),
            "project_code": project_code,
            "attachment_rules": _build_attachment_rules(row),
            "config_row_index": int(row_idx),
        }
        config.setdefault(email, []).append(entry)
    return config

EMAIL_CONFIG = load_email_config()
ALLOWED_SENDERS = set(EMAIL_CONFIG.keys())
COMPLETED_PROJECT_KEYS = load_completed_project_keys(pathlib.Path(DOWNLOAD_DIR).parent, pathlib.Path(__file__).resolve().parent)

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
        sender_cfgs = EMAIL_CONFIG.get(sender_email) or []
        if sender_cfgs:
            normalized_subj = _normalize_for_contains(text)
            for sender_cfg in sender_cfgs:
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

def derive_project_code(mail, attachment=None, *, include_other_attachments: bool = True) -> str | None:
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
    if include_other_attachments and atts:
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
    (re.compile(r"(?<!\d)(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"), ("Y","M","D")),
    # DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
    (re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})(?!\d)"), ("D","M","Y")),
    # DD-MM-YY / DD/MM/YY / DD.MM.YY
    (re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})(?!\d)"), ("D","M","Y2")),
    # YYYYMMDD (8 digits)
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), ("Y","M","D")),
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
            elif key == "Y2":
                year = _to_int(val)
                Y = 2000 + year if year is not None else None
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
def purge_previous_versions(
    download_dir: pathlib.Path,
    project_code: str,
    ext: str,
    report_date: dt.date,
    line_name: str = "",
) -> bool:
    """
    Remove older files for this project so only the newest remains.
    Returns False if an equal/newer file already exists and should be kept.
    """
    prefix = canonical_dpr_filename(project_code, report_date, ext, line_name=line_name)
    target_identity = parse_project_identity_from_filename(prefix)
    for fn in os.listdir(download_dir):
        fl = fn.lower()
        if not fl.endswith(ext.lower()):
            continue
        current_identity = parse_project_identity_from_filename(fn)
        if (
            current_identity.get("project_code", "") == target_identity.get("project_code", "")
            and current_identity.get("line_name", "") == target_identity.get("line_name", "")
        ):
            existing_date = _extract_canonical_file_date(fn, ext)
            if existing_date and existing_date >= report_date:
                return False
            try:
                os.remove(download_dir / fn)
            except Exception:
                pass
    return True


def _extract_canonical_file_date(filename: str, ext: str) -> dt.date | None:
    candidate = pathlib.Path(filename).name
    if ext and not candidate.lower().endswith(ext.lower()):
        return None
    stem = candidate[: -len(ext)] if ext else pathlib.Path(candidate).stem
    match = re.search(r"(\d{4}-\d{2}-\d{2})$", stem)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def _resolve_attachment_match(name: str, sender_cfgs: list[dict]) -> dict | None:
    normalized = _normalize_attachment_key(name, drop_digits=True)
    if not normalized:
        return None
    attachment_tokens = set(_tokenize_attachment_key(name, drop_digits=True))

    best: dict | None = None
    for entry in sender_cfgs:
        for rule_order, rule in enumerate(entry.get("attachment_rules", [])):
            match_key = rule.get("match_key") or ""
            rule_tokens = [str(t).strip() for t in (rule.get("match_tokens") or []) if str(t).strip()]
            rule_tokens = list(dict.fromkeys(rule_tokens))
            required_tokens = [t for t in rule_tokens if t not in _ATTACHMENT_TOKEN_STOPWORDS]
            if not required_tokens:
                required_tokens = rule_tokens

            substring_match = bool(match_key) and match_key in normalized
            token_match = bool(required_tokens) and all(t in attachment_tokens for t in required_tokens)
            if not substring_match and not token_match:
                continue
            match_type_rank = 2 if substring_match else 1
            match_strength = len(match_key) if substring_match else len(required_tokens)
            candidate = {
                "project_code": entry.get("project_code", "") or "",
                "line_name": rule.get("line_name", "") or "",
                "rule_matched": True,
                "match_len": match_strength,
                "match_type_rank": match_type_rank,
                "config_row_index": int(entry.get("config_row_index", 0)),
                "rule_index": int(rule_order),
            }
            if best is None:
                best = candidate
                continue
            if candidate["match_type_rank"] > best["match_type_rank"]:
                best = candidate
                continue
            if (
                candidate["match_type_rank"] == best["match_type_rank"]
                and candidate["match_len"] > best["match_len"]
            ):
                best = candidate
                continue
            if (
                candidate["match_type_rank"] == best["match_type_rank"]
                and candidate["match_len"] == best["match_len"]
            ):
                if (candidate["config_row_index"], candidate["rule_index"]) < (
                    best["config_row_index"],
                    best["rule_index"],
                ):
                    best = candidate
    if best is not None:
        return best

    # If sender has explicit attachment rules, never fallback to sender-level project
    # because that can mis-assign non-DPR attachments.
    if any(entry.get("attachment_rules") for entry in sender_cfgs):
        return None

    for entry in sender_cfgs:
        project_code = entry.get("project_code", "") or ""
        if project_code:
            return {
                "project_code": project_code,
                "line_name": "",
                "rule_matched": False,
                "match_len": 0,
                "config_row_index": int(entry.get("config_row_index", 0)),
                "rule_index": -1,
            }
    return None


def _single_sender_project(sender_cfgs: list[dict]) -> str:
    project_codes = {
        str(entry.get("project_code") or "").strip().upper()
        for entry in sender_cfgs
        if str(entry.get("project_code") or "").strip()
    }
    if len(project_codes) == 1:
        return next(iter(project_codes))
    return ""


def _attachment_is_report_like(name: str) -> bool:
    normalized = norm(name)
    if all(re.search(pat, normalized, flags=re.IGNORECASE) for pat in ATTACHMENT_MUST_CONTAIN):
        return True
    tokens = set(_tokenize_attachment_key(name, drop_digits=False))
    return "report" in tokens and ("progress" in tokens or "daily" in tokens)


def _resolve_single_project_report_fallback(name: str, sender_cfgs: list[dict]) -> dict | None:
    """
    Some single-project planners change attachment names from configured DPR text
    to variants like "TA-602 Daily Progress Report". For those senders, safely
    fall back to their configured project only when the attachment is report-like.
    Multi-project senders still require explicit attachment rules.
    """
    project_code = _single_sender_project(sender_cfgs)
    if not project_code:
        return None
    attachment_project = extract_project_code(name)
    if attachment_project and attachment_project != project_code:
        return None
    has_project_date_name = attachment_project == project_code and parse_date_from_text(name) is not None
    if not _attachment_is_report_like(name) and not has_project_date_name:
        return None
    return {
        "project_code": project_code,
        "line_name": "",
        "rule_matched": True,
        "fallback_matched": True,
        "match_len": 0,
        "match_type_rank": 0,
        "config_row_index": min(int(entry.get("config_row_index", 0)) for entry in sender_cfgs),
        "rule_index": -1,
    }

def save_latest_for_mail(mail) -> list[str]:
    saved = []
    atts = getattr(mail, "Attachments", None)
    if not atts or atts.Count == 0: return saved

    # We'll compute project per attachment (best accuracy), date once per mail (typical)
    mail_date = extract_report_date(mail, fallback_to_received=True)
    download_dir = pathlib.Path(DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)
    sender_email = get_smtp_address(mail)
    sender_cfgs = EMAIL_CONFIG.get(sender_email) or []

    for i in range(1, atts.Count + 1):
        att = atts.Item(i)
        name = att.FileName or ""
        if ALLOWED_EXTS and os.path.splitext(name)[1].lower() not in ALLOWED_EXTS:
            continue
        normalized_for_patterns = norm(name)
        resolved_match = _resolve_attachment_match(name, sender_cfgs)
        if resolved_match is None:
            resolved_match = _resolve_single_project_report_fallback(name, sender_cfgs)
        config_rule_match = resolved_match is not None and bool(resolved_match.get("rule_matched"))
        # must be a DPR-like file unless it matched a configured attachment rule
        if (not config_rule_match) and (not all(re.search(pat, normalized_for_patterns, flags=re.IGNORECASE) for pat in ATTACHMENT_MUST_CONTAIN)):
            continue

        project = ""
        line_name = ""
        if resolved_match and resolved_match.get("rule_matched"):
            project = str(resolved_match.get("project_code") or "").strip()
            line_name = normalize_line_name(resolved_match.get("line_name"))
            if not project:
                project = derive_project_code(mail, att, include_other_attachments=False) or ""
        else:
            project = derive_project_code(mail, att, include_other_attachments=False) or ""
            if not project and resolved_match:
                project = str(resolved_match.get("project_code") or "").strip()
                line_name = normalize_line_name(resolved_match.get("line_name"))
        if not project:
            # If no project code, skip (or save with original name if you prefer)
            continue
        if is_completed_project(project, COMPLETED_PROJECT_KEYS):
            logprint(f"[completed-project] skip save for {project}: marked completed")
            continue

        ext = os.path.splitext(name)[1].lower()
        # purge any previous versions for this project/ext; skip if a newer file already exists
        if not purge_previous_versions(download_dir, project, ext, mail_date, line_name=line_name):
            continue
        # save with canonical "<PROJECT> - DPR - <YYYY-MM-DD><ext>"
        target = download_dir / canonical_dpr_filename(project, mail_date, ext, line_name=line_name)
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Download DPR attachments from Outlook.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep monitoring for new mail after backfill.",
    )
    args = parser.parse_args(argv)

    pathlib.Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    ns = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    folder = get_folder_by_path(ns, FOLDER_PATH)
    logprint(f"Target folder: {folder.FolderPath}")
    logprint(f"Saving to:     {DOWNLOAD_DIR}")
    backfill(folder)
    if not args.watch:
        logprint("Done. Exiting (use --watch to keep monitoring).")
        return
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
