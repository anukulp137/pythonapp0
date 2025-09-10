import io
import re
from datetime import datetime
from dateutil import parser as dateparser

import streamlit as st
import pandas as pd

# Optional import
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# ----------- patterns -----------
CREDIT_HINTS = re.compile(r"\b(credit|credited|cr)\b", re.IGNORECASE)
DEBIT_HINTS  = re.compile(r"\b(debit|debited|dr)\b", re.IGNORECASE)
CURRENCY_CHARS = re.compile(r"[^\d\-\.\(\)]")
MONEY_PAT = re.compile(r"[-(]?\d{1,3}(?:[,]\d{3})*(?:\.\d{1,2})?[)]?")

# ----------- small utils -----------

def try_parse_date(x):
    if pd.isna(x): return None
    s = str(x).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        return dateparser.parse(s, dayfirst=True, fuzzy=True).date()
    except Exception:
        return None

def to_number(x):
    if pd.isna(x): return None
    s = str(x)
    neg = "(" in s and ")" in s
    # remove currency symbols and spaces, keep - . ( )
    s = CURRENCY_CHARS.sub("", s)
    # collapse multiple dots to one decimal point at the end
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    # handle embedded CR/DR text in the same cell
    try:
        v = float(s)
    except Exception:
        # try to pull the last money-like piece if mixed with text
        m = MONEY_PAT.findall(str(x))
        if not m:
            return None
        last = m[-1]
        last = CURRENCY_CHARS.sub("", last)
        parts = last.split(".")
        if len(parts) > 2:
            last = "".join(parts[:-1]) + "." + parts[-1]
        try:
            v = float(last)
        except Exception:
            return None
    # parentheses mean negative
    if neg and v > 0:
        v = -abs(v)
    return v

def make_unique(names):
    """Ensure column names are unique and non-empty."""
    out, seen = [], {}
    for i, raw in enumerate(names):
        name = (str(raw).strip()) if raw is not None else ""
        if not name:
            name = f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out

def coerce_series_or_first_col(df, colname):
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def pick(df, regex):
    for c in df.columns:
        if re.search(regex, c, re.IGNORECASE):
            return c
    return None

def pick_date_col(df):
    tx_date = pick(df, r"(transaction\s*date|txn\s*date|posting\s*date|\bdate\b)")
    val_date = pick(df, r"(value\s*date|val\s*date)")
    if tx_date and (not val_date or tx_date != val_date):
        return tx_date, val_date
    if val_date and not tx_date:
        return val_date, None
    return tx_date, val_date

def pick_index_col(df):
    return pick(df, r"^(s(\.|)?\s*no|sr(\.|)?\s*no|sl(\.|)?\s*no|serial|index|idx|#|id)$")

# ----------- header detection for tables -----------

HEADER_KEYWORDS = re.compile(
    r"(date|value\s*date|desc|narration|particular|details|remark|info|credit|debit|amount|cr|dr|balance)",
    re.IGNORECASE
)

def row_looks_like_header(row_cells):
    """Heuristic: if many cells contain known header words, treat as header."""
    hits = 0
    for c in row_cells:
        if c is None:
            continue
        if HEADER_KEYWORDS.search(str(c)):
            hits += 1
    return hits >= 2  # at least two header-ish cells

def align_body_to_header(body, header_len):
    """Ensure each body row matches header length by trimming/padding."""
    new_body = []
    for r in body:
        if len(r) > header_len:
            new_body.append(r[:header_len])
        elif len(r) < header_len:
            new_body.append(r + [None]*(header_len - len(r)))
        else:
            new_body.append(r)
    return new_body

# ----------- normalization -----------

def normalize_dataframe(df):
    df = df.copy()
    df.columns = make_unique(df.columns)

    # keep everything; only drop rows that are entirely empty
    df = df.dropna(how="all", axis=0)

    col_idx = pick_index_col(df)
    col_date, col_value_date = pick_date_col(df)
    col_desc = pick(df, r"(desc|narration|particular|details|remark|info)")
    col_amt  = pick(df, r"(amount(?!\s*(cr|dr))|(^|[\s_])amt($|[\s_])|transaction amount|value)")
    col_cr   = pick(df, r"(^|[\s_])credit($|[\s_])|cr amount|deposit|paid in|credit amt|cr\.?$")
    col_dr   = pick(df, r"(^|[\s_])debit($|[\s_])|dr amount|withdraw|withdrawal|paid out|debit amt|dr\.?$")
    col_type = pick(df, r"(type|dr\/cr|indicator|txn type|transaction type)")

    out = pd.DataFrame()

    # txn_id
    if col_idx:
        idx_series = coerce_series_or_first_col(df, col_idx)
        try:
            out["txn_id"] = pd.to_numeric(idx_series, errors="ignore")
        except Exception:
            out["txn_id"] = idx_series.astype(str)
    else:
        out["txn_id"] = range(1, len(df) + 1)

    # dates (allow duplicates)
    if col_date:
        ser_date = coerce_series_or_first_col(df, col_date)
        out["raw_date"] = ser_date
        out["date"] = ser_date.apply(try_parse_date)
    else:
        out["raw_date"] = None
        out["date"] = None

    if col_value_date:
        ser_vdate = coerce_series_or_first_col(df, col_value_date)
        out["value_date"] = ser_vdate.apply(try_parse_date)
    else:
        out["value_date"] = None

    # description
    if col_desc:
        out["description"] = coerce_series_or_first_col(df, col_desc).astype(str)
    else:
        exclude = {col_idx, col_date, col_value_date, col_amt, col_cr, col_dr, col_type}
        desc_cols = [c for c in df.columns if c not in exclude and df[c].notna().any()]
        out["description"] = df[desc_cols].astype(str).agg(" | ".join, axis=1) if desc_cols else ""

    # credits/debits
    if col_cr or col_dr:
        credit = coerce_series_or_first_col(df, col_cr).apply(to_number) if col_cr else pd.Series([0.0]*len(df))
        debit  = coerce_series_or_first_col(df, col_dr).apply(to_number) if col_dr else pd.Series([0.0]*len(df))
    elif col_amt and col_type:
        amt = coerce_series_or_first_col(df, col_amt).apply(to_number)
        type_series = coerce_series_or_first_col(df, col_type).astype(str)
        credit = amt.where(type_series.str.contains(r"\bcr|credit\b", case=False, regex=True), 0.0).abs()
        debit  = (-amt).where(type_series.str.contains(r"\bdr|debit\b", case=False, regex=True), 0.0).abs()
    elif col_amt:
        amt = coerce_series_or_first_col(df, col_amt).apply(to_number)
        credit = amt.where(amt >= 0.0, 0.0)
        debit  = (-amt).where(amt < 0.0, 0.0)
    else:
        credit = pd.Series([0.0]*len(df))
        debit  = pd.Series([0.0]*len(df))

    out["credit"] = pd.to_numeric(credit, errors="coerce").fillna(0.0)
    out["debit"]  = pd.to_numeric(debit, errors="coerce").fillna(0.0)

    # keep rows with any useful info
    mask_keep = (out["credit"] > 0) | (out["debit"] > 0) | out["description"].str.strip().astype(bool) | out["date"].notna()
    out = out[mask_keep].reset_index(drop=True)

    # unique txn_id even if source repeats
    if out["txn_id"].duplicated().any():
        out["txn_id"] = range(1, len(out) + 1)

    return out

# ----------- PDF parsing (robust, page-aware headers) -----------

def open_pdf(file_bytes, password: str | None):
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")
    try:
        return pdfplumber.open(io.BytesIO(file_bytes), password=(password or None))
    except Exception as e:
        msg = str(e)
        if "Incorrect password" in msg or "file has not been decrypted" in msg or "encrypted" in msg.lower():
            raise RuntimeError("This PDF appears to be encrypted. The password is missing or incorrect.")
        raise

def parse_page_tables_with_header_memory(page, page_num, last_header=None):
    """
    Extract tables; if a table has no header row, reuse last_header.
    Returns: (frames, last_header_updated)
    """
    frames = []
    tables = page.extract_tables() or []
    header_memory = last_header

    for t in tables:
        if not t or len(t) < 1:
            continue

        first_row = t[0]
        # If first row is a header, use it; else reuse previous header
        if row_looks_like_header(first_row):
            header = [str(h).strip() if h is not None else "" for h in first_row]
            header_memory = header[:]  # update memory
            body = t[1:]
        else:
            # No explicit header on this table/page — reuse last seen header if any
            if header_memory is None:
                # We cannot safely parse this table; skip it (prevents NaNs explosion)
                continue
            header = header_memory[:]
            body = t  # whole table is body

        # Align body row widths to header width
        body = align_body_to_header(body, len(header))
        df = pd.DataFrame(body, columns=make_unique(header))

        # Don't drop columns here; keep as much as possible
        df["__page__"] = page_num
        df["__mode__"] = "table"
        frames.append(df)

    return frames, header_memory

def parse_page_text(page, page_num):
    text = page.extract_text() or ""
    if not text.strip():
        return pd.DataFrame(columns=["txn_id","date","value_date","description","credit","debit","__page__","__mode__"])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    rows = []
    for ln in lines:
        amounts = [to_number(x) for x in MONEY_PAT.findall(ln)]
        vals = [a for a in amounts if a is not None]
        if not vals:
            continue
        dt = try_parse_date(ln)
        cr = dr = 0.0
        val = max(abs(a) for a in vals)
        if CREDIT_HINTS.search(ln) and not DEBIT_HINTS.search(ln):
            cr = abs(val)
        elif DEBIT_HINTS.search(ln) and not CREDIT_HINTS.search(ln):
            dr = abs(val)
        else:
            first = vals[0]
            if first >= 0: cr = abs(first)
            else:          dr = abs(first)
        rows.append({
            "txn_id": None,
            "raw_date": ln,
            "date": dt,
            "value_date": None,
            "description": ln,
            "credit": cr,
            "debit": dr,
            "__page__": page_num,
            "__mode__": "text"
        })
    if not rows:
        return pd.DataFrame(columns=["txn_id","date","value_date","description","credit","debit","__page__","__mode__"])
    return pd.DataFrame(rows)

def parse_pdf_all(file_bytes, password=None):
    """
    Parse ALL pages. For each page, try tables-with-header-memory; if no usable rows,
    fall back to text for that page. Normalize page-by-page; then concat.
    """
    with open_pdf(file_bytes, password) as pdf:
        all_norm = []
        pages = len(pdf.pages)
        rows_table = 0
        rows_text = 0
        last_header = None

        for page_num, page in enumerate(pdf.pages, start=1):
            page_frames, last_header = parse_page_tables_with_header_memory(page, page_num, last_header)

            if page_frames:
                for f in page_frames:
                    nf = normalize_dataframe(f)
                    if not nf.empty:
                        all_norm.append(nf)
                        rows_table += len(nf)
            else:
                # fallback to text on this page
                tf = parse_page_text(page, page_num)
                nf = normalize_dataframe(tf)
                if not nf.empty:
                    all_norm.append(nf)
                    rows_text += len(nf)

    if not all_norm:
        return None, {"pages": 0, "rows_table": 0, "rows_text": 0}

    big = pd.concat(all_norm, ignore_index=True, sort=False)

    meta = {
        "pages": pages,
        "rows_table": rows_table,
        "rows_text": rows_text,
    }
    return big, meta

# ----------- CSV/XLSX -----------

def parse_csv_xlsx(file_bytes, filename):
    ext = filename.lower().split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    return normalize_dataframe(df)

# ----------- summary -----------

def summarize(df):
    total_credit = df["credit"].sum()
    total_debit  = df["debit"].sum()
    counts = len(df)
    df_month = df.copy()
    df_month["month"] = df_month["date"].apply(lambda d: d.replace(day=1) if pd.notna(d) else None)
    monthly = df_month.dropna(subset=["month"]).groupby("month")[["credit","debit"]].sum().sort_index()
    return total_credit, total_debit, counts, monthly

# ----------- streamlit app -----------

def main():
    st.set_page_config(page_title="Bank Statement Analyzer", layout="wide")
    st.title("Bank Statement Analyzer")

    with st.expander("Input options", expanded=True):
        file = st.file_uploader("Upload a statement (PDF / CSV / XLSX / XLS)", type=["pdf","csv","xlsx","xls"])
        col1, col2 = st.columns([2,1])
        with col1:
            pdf_password = st.text_input("PDF password (leave blank if not encrypted)", type="password")
        with col2:
            show_preview_rows = st.number_input("Preview rows", min_value=50, max_value=1000, value=200, step=50)

    if not file:
        st.info("Upload a statement to begin.")
        return

    with st.spinner("Parsing…"):
        content = file.read()
        ext = file.name.lower().split(".")[-1]
        parsed = None
        parse_mode = ""
        meta = {"pages": 0, "rows_table": 0, "rows_text": 0}

        try:
            if ext in ("csv","xlsx","xls"):
                parsed = parse_csv_xlsx(content, file.name)
                parse_mode = "structured"
            elif ext == "pdf":
                parsed, meta = parse_pdf_all(content, password=pdf_password)
                parse_mode = "pdf (all pages; header-memory)"
            else:
                st.error("Unsupported file type.")
                return
        except Exception as e:
            st.error(f"Parsing error: {e}")
            return

    if parsed is None or parsed.empty:
        st.warning("I couldn't find transactions. If this is a scanned PDF, consider exporting CSV/XLSX or enabling OCR in a future version.")
        return

    parsed["credit"] = pd.to_numeric(parsed["credit"], errors="coerce").fillna(0.0)
    parsed["debit"]  = pd.to_numeric(parsed["debit"], errors="coerce").fillna(0.0)

    total_credit, total_debit, counts, monthly = summarize(parsed)

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total credited", f"{total_credit:,.2f}")
    c2.metric("Total debited",  f"{total_debit:,.2f}")
    c3.metric("Transactions",   f"{counts}")
    c4.metric("Parser",         parse_mode)
    c5.metric("Pages parsed",   f"{meta.get('pages',0)}")
    st.caption(f"Rows from tables: {meta.get('rows_table',0)} | Rows from text: {meta.get('rows_text',0)}")

    st.subheader("Monthly breakdown (by transaction date)")
    if not monthly.empty:
        st.dataframe(monthly.style.format("{:,.2f}"))
    else:
        st.write("No monthly grouping available (missing dates).")

    st.subheader("Preview")
    show_cols = [c for c in ["txn_id","date","value_date","description","credit","debit","__page__","__mode__"] if c in parsed.columns]
    st.dataframe(parsed[show_cols].head(int(show_preview_rows)))

    # Download normalized CSV
    csv_buf = io.StringIO()
    parsed[show_cols].to_csv(csv_buf, index=False)
    st.download_button("Download normalized CSV", data=csv_buf.getvalue(),
                       file_name="normalized_transactions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
