02 is done (full run, `QUICK_RUN = False`, T4).

**Output (public Kaggle dataset)**
- `er-02-output`: https://www.kaggle.com/datasets/xenoncipher/er-02-output

The zip has `prep/parsed_fields.parquet` (the BiLSTM-CRF fields of all 615,349 records, for 04), `models/parser_bilstm_crf.pt` (for 08), `parser_eval.csv` and `parser_meta.json`.

**Setup.** The silver labels come from the rule parser, run on the addresses *with* their commas. The BiLSTM-CRF (word + character embeddings, 2.72M parameters) is trained on 120,000 train addresses for 3 epochs (1.4 min). The commas are removed from half of them, so the model has to find the field boundaries without the punctuation. Evaluation is on val. The rules were fixed in #22: they now find states in any position (42 state spellings learned on train), skip null markers, and tag leading unit/house tokens as HOUSE.

### Results (`parser_eval.csv`, val, against the silver labels)

| input | parser | token accuracy | macro F1 | exact address |
|---|---|---|---|---|
| with commas | rules | 1.0000 | 1.0000 | 1.0000 |
| with commas | BiLSTM-CRF | 0.9994 | 0.9989 | 0.9969 |
| **commas removed** | rules | 0.5758 | 0.3691 | **0.0000** |
| **commas removed** | **BiLSTM-CRF** | **0.9500** | **0.9063** | **0.8545** |

- *With commas* the rules score 1.0 by construction, because they produce the silver labels. The BiLSTM-CRF reproduces them almost exactly (0.9994).
- *Commas removed* is the real test. The rules need commas to split the fields, so without them every address is wrong (they put everything after the house number into one field). The BiLSTM-CRF still gets 85% of addresses exactly right.
- Caveat: **no real address in er-small lacks commas.** 1,636 of 61,516 val records have no comma, and all of them are empty. The comma-free condition is a robustness test (commas removed on purpose), not a case from this data. That is also why the BiLSTM-CRF fields do not change 04 (see #5).

### Per-tag report (BiLSTM-CRF, commas removed, val)

| tag | precision | recall | F1 | support (tokens) |
|---|---|---|---|---|
| HOUSE | 0.979 | 0.978 | 0.979 | 13,027 |
| STREET | 0.893 | 0.950 | 0.920 | 12,330 |
| LOCALITY | 0.944 | 0.916 | 0.930 | 24,450 |
| LANDMARK | 0.517 | 0.836 | 0.639 | 831 |
| CITY | 0.993 | 0.951 | 0.972 | 13,265 |
| STATE | 0.999 | 0.998 | 0.999 | 11,296 |
| macro avg | 0.887 | 0.938 | 0.906 | 75,199 |

**POSTCODE is missing because er-small has no postcodes.** The rule parser found none (`p_postcode` is empty for all 615,349 records), so the tag has no support. Macro F1 is over the six tags that occur.

- **STATE / CITY / HOUSE are easy:** STATE comes from a small closed vocabulary, CITY sits just before it, and HOUSE is the leading run of tokens with digits.
- **STREET vs LOCALITY is the hard boundary.** Without commas, `abis arcade pattambi road` can be one street or a building + street, and the silver labels split it only where the comma was.
- **LANDMARK has low precision but high recall.** The rules tag LANDMARK only when a component *starts* with near / opp / behind / infront…. The BiLSTM-CRF also finds landmark phrases in the middle of a component (see the examples below). Part of its "false positives" are real landmarks that the silver labels miss.

### 5 addresses where the BiLSTM-CRF fixes the rules (commas removed)

The BiLSTM-CRF matches the silver parse exactly on 17,116 of 20,000 val addresses (≥ 6 tokens) without commas. Five of them:

1. `003Rd Floor Platinum Plaza Plot No.24, Pu-4, Behind C-21 Mall, A.b. Road, Indore, MP`: all six fields
   - BiLSTM-CRF: HOUSE `003rd floor` · LOCALITY `platinum plaza plot no.24 pu-4` · LANDMARK `behind c-21 mall` · STREET `a.b. road` · CITY `indore` · STATE `mp`
   - rules: HOUSE `003rd floor` · STREET `platinum plaza … a.b. road indore mp`
2. `1915 Claybourne Avenue, Salt Lake City, Utah`: multi-word city
   - BiLSTM-CRF: HOUSE `1915` · STREET `claybourne avenue` · CITY `salt lake city` · STATE `utah`
   - rules: HOUSE `1915` · STREET `claybourne avenue salt lake city utah`
3. `Happy Valley, 17751. Vogel Court, Oregon`: shuffled components, city first
   - BiLSTM-CRF: CITY `happy valley` · HOUSE `17751.` · STREET `vogel court` · STATE `oregon`
   - rules: STREET `happy valley 17751. vogel court oregon`
4. `#87 C/o E K Viswambharan, KL, Thrissur, Elavanaparambil House, Puthenchira North, Thrissur`: state in the middle
   - BiLSTM-CRF: HOUSE `#87` · LOCALITY `c/o e k viswambharan thrissur elavanaparambil house puthenchira north` · STATE `kl` · CITY `thrissur`
   - rules: HOUSE `#87` · LOCALITY `c/o … kl thrissur … thrissur`
5. `Bldgno.2351b, Iyyappadi Housecivilstationpo, Calicut, Kozhikode, കേരളം`: state in Malayalam script (romanised `kerlm`)
   - BiLSTM-CRF: HOUSE `bldgno.2351b` · LOCALITY `iyyappadi housecivilstationpo calicut` · CITY `kozhikode` · STATE `kerlm`
   - rules: HOUSE `bldgno.2351b` · LOCALITY `iyyappadi … kozhikode kerlm`

### 3 addresses where both fail (commas removed)

1. `WARD NO. 10, SHUJALPUR ROAD INFRONT OF MARKETING SOCIETY, AKODIA, Madhya Pradesh`
   - silver: STREET `shujalpur road infront of marketing society`
   - BiLSTM-CRF: STREET `shujalpur road` · LANDMARK `infront of marketing society`
   - It "fails" only against the silver label. The model's parse is the better one: the landmark is in the middle of a component, where the rules don't look.
2. `59/493 Kadungamparambil House, Chalanchery Road, Kochi, Ernakulam, Kerala`
   - silver: HOUSE `59/493 kadungamparambil house` · STREET `chalanchery road`
   - BiLSTM-CRF: HOUSE `59/493` · STREET `kadungamparambil house chalanchery road`
   - A real boundary error. In Kerala, `<family name> House` is part of the house address, but without the comma the model attaches it to the following street.
3. `Ernakulam, Keralam, Kochi, Nh-47 Bypass, Kundanoor Junction, Maradu, Amv Towers`
   - silver: CITY `amv towers` · STATE `keralam`
   - BiLSTM-CRF: STREET `kochi nh-47 bypass kundanoor junction` · LOCALITY `ernakulam maradu amv towers` · no CITY
   - A shuffled address where even the silver label is wrong: the rules take the last component without digits as the city. The model gets the state right but finds no city at all. Shuffled components (10–14% of addresses) are the main limit of the silver labels.

All 25 + 25 candidates are in `analysis/out_parser/parser_examples.md`. It is regenerated by `analysis/parser_examples.py` from the saved model.
