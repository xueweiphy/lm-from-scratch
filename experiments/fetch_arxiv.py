#!/usr/bin/env python3
"""
Build a TinyStories-style corpus from arXiv abstracts.

Output is one text file in exactly the layout of TinyStoriesV2-GPT4-*.txt:
documents one after another, each followed by a <|endoftext|> line, so the
same BPE tokenizer, encode_datasets.py, and data loader work unchanged.
Each document is "title\n\nabstract" (drop titles with --no-titles), and the
records are shuffled and split 90/10 into <out>_train.txt and <out>_valid.txt.

Cleaning is lighter than the char-level version this grew from: HTML entities,
typographic punctuation and Greek letters are normalised (Greek -> LaTeX, which
the abstracts already use), hard wraps are joined -- but no character is
dropped for being rare; byte-level BPE handles the long tail.

Standard library only.

    python3 experiments/fetch_arxiv.py                          # hep-ph, ~10M chars -> data/arxiv_hepph_{train,valid}.txt
    python3 experiments/fetch_arxiv.py --category astro-ph.CO   # a different corner of arXiv
    python3 experiments/fetch_arxiv.py --target-chars 30000000  # bigger (API stops at ~30k results)
    python3 experiments/fetch_arxiv.py --self-test              # check parsing/cleaning offline

Please leave --delay at 3 seconds or more: that is the interval arXiv asks API
clients to respect, and this script is polite by default.

  Wei Xue, August 2026; TinyStories layout September 2026.
"""

import argparse, html, os, random, re, sys, time, unicodedata
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
END_OF_TEXT = "<|endoftext|>"
ATOM = "{http://www.w3.org/2005/Atom}"


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

# Characters that turn up constantly in physics abstracts and would otherwise
# each become their own rare token in a char-level vocabulary.
REPLACEMENTS = {
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '−': '-', '‐': '-', '‑': '-',
    ' ': ' ', ' ': ' ', ' ': ' ', ' ': ' ', '​': '',
    '…': '...', '×': 'x', '·': '.', '′': "'",
    '→': ' -> ', '←': ' <- ', '⇒': ' => ', '↔': ' <-> ',
    '≤': ' <= ', '≥': ' >= ', '≈': ' ~ ', '≡': ' = ',
    '±': ' +/- ', '≃': ' ~ ', '∼': ' ~ ', '≪': ' << ',
    '≫': ' >> ', '≠': ' != ', '∞': ' infinity ',
    '†': '+', '∗': '*', '⋅': '.', 'º': ' deg ',
    '°': ' deg ', '″': '"', '˜': '~', '­': '',
    '∈': ' in ', '∉': ' notin ', '⊂': ' subset ', '∪': ' union ',
    '∩': ' cap ', '∀': ' forall ', '∃': ' exists ', '∇': ' nabla ',
    '∂': ' partial ', '√': ' sqrt ', '∑': ' sum ', '∏': ' prod ',
    '∫': ' int ', '⟨': '<', '⟩': '>', '〈': '<', '〉': '>',
    '‖': '||', '∼': ' ~ ', '≲': ' <~ ', '≳': ' >~ ',
    '⊗': ' x ', '⊕': ' + ', '∅': ' empty ', '∝': ' propto ',
    'ℏ': ' hbar ', 'ℓ': 'l', '⟶': ' -> ', '％': '%',
}

# Greek letters -> LaTeX names. hep-ph abstracts are full of these, and the
# LaTeX spelling is already present elsewhere in the same corpus, so mapping
# to it makes the text more self-consistent rather than less.
GREEK = {
    'α': r'\alpha ',  'β': r'\beta ',   'γ': r'\gamma ',
    'δ': r'\delta ',  'ε': r'\epsilon ', 'ζ': r'\zeta ',
    'η': r'\eta ',    'θ': r'\theta ',  'ι': r'\iota ',
    'κ': r'\kappa ',  'λ': r'\lambda ', 'μ': r'\mu ',
    'ν': r'\nu ',     'ξ': r'\xi ',     'π': r'\pi ',
    'ρ': r'\rho ',    'σ': r'\sigma ',  'τ': r'\tau ',
    'υ': r'\upsilon ', 'φ': r'\phi ',   'χ': r'\chi ',
    'ψ': r'\psi ',    'ω': r'\omega ',
    'Γ': r'\Gamma ',  'Δ': r'\Delta ',  'Θ': r'\Theta ',
    'Λ': r'\Lambda ', 'Ξ': r'\Xi ',     'Π': r'\Pi ',
    'Σ': r'\Sigma ',  'Φ': r'\Phi ',    'Ψ': r'\Psi ',
    'Ω': r'\Omega ',
}


def clean(text):
    """Normalise one abstract into a compact, low-vocabulary ASCII stream."""
    text = html.unescape(text)
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    for k, v in GREEK.items():
        text = text.replace(k, v)

    # strip accents (résumé -> resume) rather than keeping one-off code points
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # anything still outside ASCII becomes a space rather than a one-off token
    text = ''.join(c if ord(c) < 128 else ' ' for c in text)

    # our Greek mapping appends a space; undo it before a sub/superscript so
    # "\alpha _s" reads as "\alpha_s" the way it was written
    text = re.sub(r'\\([a-zA-Z]+) ([_^{/,.;:)\]}])', r'\\\1\2', text)

    # arXiv hard-wraps abstracts; join them back into paragraphs
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def parse_atom(xml_bytes):
    """Return [(title, abstract), ...] from one arXiv API response."""
    root = ET.fromstring(xml_bytes)
    out = []
    for entry in root.findall(ATOM + 'entry'):
        summary = entry.find(ATOM + 'summary')
        title = entry.find(ATOM + 'title')
        if summary is None or summary.text is None:
            continue
        t = title.text if (title is not None and title.text) else ''
        out.append((t, summary.text))
    return out


def fetch_page(category, start, per_page, delay, retries=3, year=None):
    query = f'cat:{category}'
    if year is not None :                       # the API paginates to 10,000 results per query,
        query += f' AND submittedDate:[{year}01010000 TO {year}12312359]'   # so slice by year to get past it
    q = urllib.parse.urlencode({
        'search_query': query,
        'start': start,
        'max_results': per_page,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    })
    url = f"{API}?{q}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'char-gpt-corpus-builder/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return parse_atom(r.read())
        except Exception as e:
            wait = delay * (attempt + 2)
            print(f"[warn] {type(e).__name__}: {e} -- retrying in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    return []


def to_corpus(records, titles=True):
    """Lay documents out the TinyStories way: each one ends with an <|endoftext|> line."""
    docs = [f"{title}\n\n{abstract}" if (titles and title) else abstract
            for title, abstract in records]
    return "".join(f"{doc}\n{END_OF_TEXT}\n" for doc in docs)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--category',     default='hep-ph')
    p.add_argument('--year',         type=int, default=None,
                   help='restrict to one submission year; the stem gets _<year>. '
                        'Use this to get past the 10,000-results-per-query cap')
    p.add_argument('--out',          default=None,
                   help='output stem; default data/arxiv_<category>. '
                        'Writes <stem>_train.txt and <stem>_valid.txt')
    p.add_argument('--target-chars', type=int, default=10_000_000,
                   help='stop once the corpus reaches this size')
    p.add_argument('--per-page',     type=int, default=200, help='<= 200')
    p.add_argument('--max-pages',    type=int, default=150)
    p.add_argument('--valid-frac',   type=float, default=0.10,
                   help='fraction of documents held out as the validation file')
    p.add_argument('--delay',        type=float, default=3.0,
                   help="seconds between API calls; arXiv asks for >= 3")
    p.add_argument('--min-chars',    type=int, default=300,
                   help='skip abstracts shorter than this')
    p.add_argument('--no-titles',    action='store_true',
                   help='abstract only; default is "title\\n\\nabstract"')
    p.add_argument('--no-shuffle',   action='store_true',
                   help='keep reverse-chronological order (default shuffles, so '
                        'the 80/20 train/val split is not a split in time)')
    p.add_argument('--seed',         type=int, default=1337)
    p.add_argument('--self-test',    action='store_true',
                   help='run the parser and cleaner on a built-in sample, no network')
    args = p.parse_args()

    if args.self_test:
        return self_test()

    print(f"[fetch] category {args.category}{' ' + str(args.year) if args.year else ''}, target {args.target_chars:,} chars")
    seen, records, total = set(), [], 0

    for page in range(args.max_pages):
        start = page * args.per_page
        batch = fetch_page(args.category, start, args.per_page, args.delay, year=args.year)
        if not batch:
            print(f"[fetch] no results at start={start}; stopping")
            break

        kept = 0
        for title, summary in batch:
            a = clean(summary)
            if len(a) < args.min_chars:
                continue
            key = a[:120]
            if key in seen:
                continue
            seen.add(key)
            records.append((clean(title), a))
            total += len(a) + 2
            kept += 1

        print(f"[fetch] start={start:5d}  got {len(batch):3d}  kept {kept:3d}  "
              f"total {total:,} chars")
        if total >= args.target_chars:
            break
        time.sleep(args.delay)

    if not records:
        print("[error] nothing fetched. Check network access and try again.",
              file=sys.stderr)
        return 1

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)

    n_valid = max(1, int(len(records) * args.valid_frac))
    stem = args.out or os.path.join(DATA_DIR, f"arxiv_{args.category.replace('.', '_').replace('-', '')}")
    if args.year and not args.out :
        stem += f"_{args.year}"
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for split, subset in (("valid", records[:n_valid]), ("train", records[n_valid:])):
        text = to_corpus(subset, titles=not args.no_titles)
        path = f"{stem}_{split}.txt"
        with open(path, 'w') as f:
            f.write(text)
        report(text, len(subset), path)
    return 0


def report(text, n_docs, path):
    print("-" * 66)
    print(f"[done] {path}")
    print(f"  documents     : {n_docs:,}   ({text.count(END_OF_TEXT):,} {END_OF_TEXT} markers)")
    print(f"  characters    : {len(text):,}")
    print(f"  chars/document: {len(text)//max(n_docs,1):,}")
    print("-" * 66)


SAMPLE_ATOM = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Probing the Higgs self-coupling</title>
    <summary>  We study the di-Higgs production cross section at
the LHC, including next-to-leading order corrections of order
&#945;_s^3. The trilinear coupling &#955; is constrained to
&#955;/&#955;_SM &#8712; [-1.5, 6.7] at 95% CL, and we find
&#963; = 32.7 &#177; 2.1 fb &#8212; a 30% improvement.
</summary>
  </entry>
  <entry>
    <title>Short one</title>
    <summary>Too short.</summary>
  </entry>
</feed>'''


def self_test():
    """Exercise the parse + clean path with no network."""
    entries = parse_atom(SAMPLE_ATOM.encode())
    assert len(entries) == 2, entries
    title, summary = entries[0]
    out = clean(summary)
    print("[self-test] cleaned abstract:")
    print("   ", out)
    checks = {
        'no newlines':        '\n' not in out,
        'greek -> latex':     r'\alpha' in out and r'\lambda' in out,
        'plusminus expanded': '+/-' in out,
        'emdash -> hyphen':   '—' not in out,
        'entities decoded':   '&#' not in out and '&amp;' not in out,
        'ascii only':         all(ord(c) < 128 for c in out),
        'no double spaces':   '  ' not in out,
    }
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    body = to_corpus([(clean(t), clean(s)) for t, s in entries])
    checks['tinystories layout'] = body.count(END_OF_TEXT) == 2 and body.endswith(END_OF_TEXT + "\n")
    print(f"   {'PASS' if checks['tinystories layout'] else 'FAIL'}  tinystories layout")
    print("[self-test] corpus:")
    print(body)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
