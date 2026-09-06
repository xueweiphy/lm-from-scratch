# Tokenizer experiments

Compression measurements for the two trained BPE tokenizers, covering
parts (a) and (b) of the `tokenizer_experiments` problem (assignment 1,
section 2.7).

Run with `python tokenizer_experiments.py` from this directory.

## Method

A **document** is one `<|endoftext|>`-delimited chunk of a validation
file. Ten documents are drawn from each corpus without replacement,
`random.Random(42).sample`.

The **compression ratio** is aggregate: total UTF-8 bytes divided by
total tokens over the whole sample. This is not the same as the mean of
the per-document ratios, and the aggregate is what "bytes/token"
conventionally means, since it weights long documents proportionally.

`tokens/pretoken` is reported alongside it as a diagnostic. Because BPE
merges never cross a pretoken boundary, every pretoken yields at least
one token, so this quantity is bounded below by 1.0. It measures how
often the tokenizer is forced to break a word apart, isolating the
tokenizer's vocabulary from the properties of the text.

## Results

| sample | tokenizer | bytes | pretokens | tokens | bytes/token | tokens/pretoken |
|---|---|---|---|---|---|---|
| TinyStories | TinyStories 10K | 8482 | 2100 | 2107 | 4.026 | 1.003 |
| OpenWebText | OpenWebText 32K | 39375 | 8482 | 9275 | 4.245 | 1.093 |
| OpenWebText | TinyStories 10K | 39375 | 8482 | 12439 | 3.165 | 1.467 |
| TinyStories | OpenWebText 32K | 8482 | 2100 | 2170 | 3.909 | 1.033 |

Rows 1-2 answer part (a); row 3 answers part (b); row 4 is the reverse
of (b), included because the comparison is the point.

## Findings

**Each tokenizer compresses its own corpus to roughly 4 bytes/token.**
For TinyStories this is nearly trivial: at 1.003 tokens/pretoken the
tokenizer splits 7 words out of 2100, so the ratio reduces to little
more than the average byte-length of a TinyStories word. A 10K
vocabulary is enough to memorize essentially all of a children's-story
lexicon.

**Cross-tokenizing degrades compression, and the degradation is
asymmetric.** OpenWebText through the TinyStories tokenizer loses 25%
(4.245 -> 3.165); TinyStories through the OpenWebText tokenizer loses
only 3% (4.026 -> 3.909).

The cause is vocabulary coverage, not under-training. TinyStories'
lexicon is close to a subset of OpenWebText's — simple English words are
frequent on the web too — so the 32K tokenizer already holds nearly all
of them. The containment does not run the other way: no amount of
additional training on TinyStories would produce a merge for `Rohingya`
or `UNHCR`, because those strings never occur there.

The words each tokenizer fails on show this directly. OpenWebText
pretokens that the TinyStories tokenizer splits into three or more
pieces are proper nouns and rare terms — ` Aadhaar`, ` Rohingya`,
` Jammu`, ` UNHCR`, ` Hyderabad`, `Flaherty`, ` eToro`. In the reverse
direction only 32 pretoken types split more under the OpenWebText
tokenizer, and they are TinyStories' own cast and idiom:

    ' Lily'     -> ' L' + 'ily'
    ' Sunny'    -> ' Sun' + 'ny'
    ' bossy'    -> ' boss' + 'y'
    ' Mia'      -> ' M' + 'ia'
    ' mustache' -> ' must' + 'ache'

The TinyStories tokenizer spent its 10K slots on exactly these; the
OpenWebText tokenizer spread 32K across the whole web and cannot afford
a dedicated token for ` bossy`.

**General principle.** A tokenizer's compression ratio measures how well
its training distribution matches the text it is applied to, and the
penalty is asymmetric: a broad tokenizer degrades gracefully on narrow
text, a narrow one degrades sharply on broad text.

## Caveats

Ten documents per corpus is a small sample and the ratios carry real
sampling noise; the honest summary of part (a) is "about 4 bytes/token
for each tokenizer on its own corpus" rather than three significant
figures. Changing the seed, or sampling with a different RNG, moves the
TinyStories figure by a few percent — the notebook version of this
experiment used `torch.randint` with seed 42 and obtained 4.145 on a
different draw of ten documents.

Note also that 93% of OpenWebText pretokens are already single tokens
under its own 32K vocabulary. The average of 1.093 comes from a small
tail: 475 pretokens split into 2 tokens, 121 into 3, 24 into 4, 1 into
5. Roughly a quarter of all pretokens are 2 bytes or fewer — mostly
single punctuation marks, which never split — which pulls the average
toward 1 and is why this number sits below the ~1.3 tokens/word often
quoted for GPT-2.
