"""Build an arXiv corpus from the Kaggle metadata dump — every abstract in a category,
without 30 hours of API calls (the API paginates to 10,000 results per query).

Download once (~4 GB, JSON-lines, one paper per line):
    https://www.kaggle.com/datasets/Cornell-University/arxiv
and put arxiv-metadata-oai-snapshot.json in data/.

    python experiments/arxiv_from_kaggle.py                        # hep-ph
    python experiments/arxiv_from_kaggle.py hep-th gr-qc           # several categories, one corpus
    python experiments/arxiv_from_kaggle.py hep-ph --from-year 2000

Same layout as fetch_arxiv.py: "title\\n\\nabstract" documents each followed by an
<|endoftext|> line, shuffled, split 90/10, so the same tokenizer and loader work.
"""
import json, os, random, sys

HERE = os.path.dirname ( os.path.abspath ( __file__ ) )
sys.path.insert ( 0, HERE )
from fetch_arxiv import clean, to_corpus, report, DATA_DIR

Dump = os.path.join ( DATA_DIR, "arxiv-metadata-oai-snapshot.json" )
Min_chars, Valid_frac, Seed = 300, 0.10, 1337


def main () :
    args = [ a for a in sys.argv[1:] if not a.startswith ( "--" ) ] or [ "hep-ph" ]
    from_year = next ( ( int ( a.split("=")[1] ) for a in sys.argv[1:] if a.startswith ( "--from-year" ) ), 0 )
    wanted = set ( args )

    records = []
    with open ( Dump ) as file :
        for line in file :
            paper = json.loads ( line )
            if not wanted & set ( paper.get ( "categories", "" ).split() ) :
                continue
            if from_year and int ( paper.get ( "update_date", "0000" ) [:4] ) < from_year :
                continue
            abstract = clean ( paper.get ( "abstract", "" ) )
            if len ( abstract ) >= Min_chars :
                records.append ( ( clean ( paper.get ( "title", "" ) ), abstract ) )

    print ( f"[kaggle] {' '.join(sorted(wanted))}: {len(records):,} papers" )
    random.Random ( Seed ).shuffle ( records )
    n_valid = max ( 1, int ( len ( records ) * Valid_frac ) )

    stem = os.path.join ( DATA_DIR, "arxiv_" + "_".join ( sorted ( w.replace(".","_").replace("-","") for w in wanted ) ) + "_full" )
    for split, subset in ( ( "valid", records[:n_valid] ), ( "train", records[n_valid:] ) ) :
        text = to_corpus ( subset )
        path = f"{stem}_{split}.txt"
        with open ( path, "w" ) as file :
            file.write ( text )
        report ( text, len ( subset ), path )


if __name__ == "__main__" :
    main()
