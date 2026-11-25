---
name: cof-monomer-extraction
description: Identify aldehyde, amine, and hydrazide monomers in COF papers and look up their connectivity SMILES in PubChem.
---

# COF Monomer Extraction

Read each Markdown paper to identify its title and the full names of the aldehyde and amine or hydrazide monomers in each COF. Use PubChem to retrieve their connectivity SMILES.

## Usage

Run from the skill directory. List one compound per line in `names.txt` for file input.

```bash
python -m pip install requests
python scripts/pubchem_smiles.py --json "terephthalaldehyde"
python scripts/pubchem_smiles.py --json --file names.txt
```

Write one CSV row per COF with the columns `aldehyde,SMILES1,amine,SMILES2,title`. Leave unresolved SMILES fields empty.
