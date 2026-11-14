# Report section: Address parsing

**Author:** Keshav (issues #4, #6) · **Notebook:** `02_parser_bilstm_crf` · **Analysis:** `analysis/parser_examples.py`
**Course topics:** parsing (sequence labelling), LSTM, conditional random fields, weak supervision

Tables: `parser_eval.csv` and `parser_meta.json` in the `er-02-output` Kaggle dataset; all examples in
`analysis/out_parser/parser_examples.md` (regenerated from the saved model by `analysis/parser_examples.py`).
Output used downstream: `prep/parsed_fields.parquet` (the fields of all 615,349 records) → `04_gbm_baseline`,
and the saved model → `08_demo`.

---

## P.1 Why parse addresses

Two branches of one business chain often have the same name and differ only in the house number or the street,
so a matcher should compare addresses **field by field**, not as one string. A parser turns an address into
labelled spans:

```
003rd floor | platinum plaza plot no.24 pu-4 | behind c-21 mall | a.b. road | indore | mp
   HOUSE    |          LOCALITY              |     LANDMARK     |   STREET  |  CITY  | STATE
```

Tag set: `HOUSE`, `STREET`, `LOCALITY`, `LANDMARK`, `CITY`, `STATE`, `POSTCODE`, plus `SEP` for a comma token.
This is sequence labelling (shallow parsing): one tag per token, with the fields as maximal runs of one tag.

Tokens come from the shared address normalisation: NFKC, romanisation of non-Latin scripts (so `കേരളം` becomes
`kerlm`), lower-casing, splitting on whitespace and commas, dropping null markers (`null`, `n/a`), at most 40 tokens.

## P.2 Silver labels from rules

There are **no gold address parses** in the data, so the labels are produced by a rule parser that relies on the
**commas** between address components:

1. **Postcode and state at the end.** Postcodes are 5- or 6-digit tokens (`pin` / `zip` prefixes included).
   State spellings are *learned on train*: a component counts as a state if it occurs ≥ 30 times and is the last
   component in ≥ 60 % of its occurrences. This gives 42 spellings (`kerala`, `kl`, `keralam`, `kerlm`, `or`,
   `oregon`, `ut`, `mp`, `madhya pradesh`, …). The threshold is 0.6, not 0.9, because 10–14 % of addresses have
   shuffled components (`61022 Holly Court, OR, Bend`); city names stay below 0.45.
2. **City:** the last component before the trailing state/postcode that has no digits and is not a state.
3. **House:** leading tokens with digits or house words (`no.`, `plot`, `door`, `#`, `flat`, …) in the first
   component, or in any component that starts with a unit word (`suite no.889 …`). A first component ending in a
   Kerala house name (`… house`, `bhavan`, `villa`) is all HOUSE.
4. **Other components:** `LANDMARK` if it starts with *near / opp / behind / infront …*, `STREET` if it contains a
   street word (*road, street, lane, marg, salai …*), otherwise `LOCALITY`. A state found in the middle (shuffled
   address) is still tagged `STATE`.

With commas, the rules are a reasonable parser. Without them, they cannot find the component boundaries at all.

## P.3 Model: BiLSTM-CRF

```
token t ──► word embedding (100)            ─┐
token t ──► chars ──► char BiLSTM (2 × 32)  ─┴► concat (164) ──► dropout 0.2
        ──► 2-layer BiLSTM (2 × 128 per layer, dropout 0.2) ──► linear ──► emission scores (8 tags)
        ──► CRF (8 × 8 transition scores) ──► Viterbi decoding
```

- **Word embedding:** the 30,000 most frequent training tokens seen ≥ 3 times, with digits mapped to `0`
  (`plot 24` and `plot 37` share an embedding). Other words map to an unknown token.
- **Character BiLSTM** over the first 16 characters of each token (characters seen ≥ 5 times). It covers what the
  word vocabulary does not: unseen house numbers (`59/493`, `bldgno.2351b`), romanised spelling variants, and
  suffixes like `-nagar`, `-puram`.
- **2-layer BiLSTM:** each tag depends on context in both directions. `road` usually ends a street, but in
  `near railway road` it is part of a landmark.
- **CRF output layer:** instead of choosing each tag independently, the CRF scores whole tag sequences with learned
  transition scores (e.g. `CITY → STATE` likely, `STATE → HOUSE` rare). Training maximises the log-likelihood of the
  correct sequence (forward algorithm). Decoding finds the best sequence with Viterbi.

2.72M parameters in total.

**Training trick: remove the commas.** The silver labels are computed on the address *with* commas. Then, for
**half** of the training addresses, the comma tokens are deleted from both the tokens and the tags. The model therefore
has to learn the field boundaries from the words themselves: street words, state names, the position of numbers. This
is the one thing the rules cannot do, since the rules *are* the comma structure.

**Setup.** 120,000 random training addresses (non-empty), 3 epochs, batch 256, Adam (lr 2e-3), gradient clipping at
5.0, Kaggle T4: **1.4 min**. Evaluation on 10,000 random non-empty val addresses.

## P.4 Results

Against the silver labels on val (`parser_eval.csv`). Macro F1 is over the six tags that occur (see P.6 for POSTCODE).

| input | parser | token accuracy | macro F1 | exact address |
|---|---|---|---|---|
| with commas | rules | 1.0000 | 1.0000 | 1.0000 |
| with commas | BiLSTM-CRF | 0.9994 | 0.9989 | 0.9969 |
| **commas removed** | rules | 0.5758 | 0.3691 | **0.0000** |
| **commas removed** | **BiLSTM-CRF** | **0.9500** | **0.9063** | **0.8545** |

- **With commas** the rules score 1.0 by construction, because they produce the labels. The BiLSTM-CRF reproduces
  them almost exactly (0.9994 token accuracy, 99.7 % of addresses exactly right).
- **Without commas** is the real test. The rules see one component, so after the house number they put everything into
  one field and get **no** address fully right. The BiLSTM-CRF still parses **85 %** of addresses exactly and 95 % of
  tokens correctly.

**Per tag (BiLSTM-CRF, commas removed, val):**

| tag | precision | recall | F1 | support (tokens) |
|---|---|---|---|---|
| HOUSE | 0.979 | 0.978 | 0.979 | 13,027 |
| STREET | 0.893 | 0.950 | 0.920 | 12,330 |
| LOCALITY | 0.944 | 0.916 | 0.930 | 24,450 |
| LANDMARK | 0.517 | 0.836 | 0.639 | 831 |
| CITY | 0.993 | 0.951 | 0.972 | 13,265 |
| STATE | 0.999 | 0.998 | 0.999 | 11,296 |
| macro avg | 0.887 | 0.938 | 0.906 | 75,199 |

- **STATE, CITY and HOUSE are easy.** STATE comes from a small closed vocabulary, CITY sits just before it, and HOUSE is
  the leading run of tokens with digits.
- **STREET vs LOCALITY is the hard boundary.** Without commas, `abis arcade pattambi road` can be one street or a
  building plus a street; the silver labels split it only where the comma was.
- **LANDMARK has low precision but high recall.** The rules tag LANDMARK only when a component *starts* with a landmark
  word. The BiLSTM-CRF also finds landmark phrases in the middle of a component (P.5, example 4). Part of its "false
  positives" are real landmarks that the silver labels miss.

## P.5 Examples (commas removed)

The BiLSTM-CRF matches the silver parse exactly on 17,116 of 20,000 val addresses with ≥ 6 tokens.

| # | address (shown with its commas) | BiLSTM-CRF without commas | rules without commas |
|---|---|---|---|
| 1 | `1915 Claybourne Avenue, Salt Lake City, Utah` | HOUSE `1915` · STREET `claybourne avenue` · CITY `salt lake city` · STATE `utah` | HOUSE `1915` · STREET `claybourne avenue salt lake city utah` |
| 2 | `Happy Valley, 17751. Vogel Court, Oregon` (shuffled) | CITY `happy valley` · HOUSE `17751.` · STREET `vogel court` · STATE `oregon` | STREET `happy valley 17751. vogel court oregon` |
| 3 | `Bldgno.2351b, Iyyappadi Housecivilstationpo, Calicut, Kozhikode, കേരളം` | HOUSE `bldgno.2351b` · LOCALITY `iyyappadi housecivilstationpo calicut` · CITY `kozhikode` · STATE `kerlm` | HOUSE `bldgno.2351b` · LOCALITY `iyyappadi … kozhikode kerlm` |
| 4 | `WARD NO. 10, SHUJALPUR ROAD INFRONT OF MARKETING SOCIETY, AKODIA, Madhya Pradesh` | STREET `shujalpur road` · LANDMARK `infront of marketing society` | (silver label: STREET `shujalpur road infront of marketing society`) |
| 5 | `59/493 Kadungamparambil House, Chalanchery Road, Kochi, Ernakulam, Kerala` | HOUSE `59/493` · STREET `kadungamparambil house chalanchery road` | (silver label: HOUSE `59/493 kadungamparambil house`) |

- Examples 1–3: the model finds a multi-word city, a shuffled component order, and a state written in Malayalam script,
  with no punctuation.
- Example 4 counts as an error against the silver label, but the model's parse is better: the landmark is in the middle
  of a component, where the rules don't look.
- Example 5 is a real boundary error. In Kerala, `<family name> House` belongs to the house address; without the comma
  the model attaches it to the street that follows.

## P.6 Limitations

- **Silver labels, not gold.** The parser is evaluated against the rules that trained it, so the numbers measure how
  well it *recovers the comma structure*, not how well it parses. Rule errors propagate. The biggest source is shuffled
  components (10–14 % of addresses), where the rules can take a building name as the city.
- **No postcodes.** None of the 615,349 records in our subset has a postcode the rules recognise (`p_postcode` is empty
  everywhere), so the POSTCODE tag has no support and is left out of macro F1.
- **The robustness test is synthetic.** Every non-empty address in our data has its commas (the 1,636 of 61,516 val
  records without a comma all have an empty address). Comma-free input is a stress test we created, not a case from
  this data.

## P.7 Does the parser help matching?

Not measurably on this data. The LightGBM baseline (Section G.5) was run twice, with the BiLSTM-CRF fields and with
the rule fields. The difference is +0.0003 test F0.5 and **not significant** (paired bootstrap interval
[−0.00006, +0.00064]). Since the addresses have their commas, the two parsers produce almost the same fields
(99.7 % of addresses identical). On top of that, the parsed-field features carry only 0.6 % of the GBM's gain.

The parser's value is **robustness**: on addresses without punctuation (OCR output, form fields joined by spaces,
other sources), the rules fail completely, and the BiLSTM-CRF keeps 85 % of addresses exactly right. It is also used
by the Ditto matcher (06g) to mark address spans, and the live demo (08) shows its fields.
