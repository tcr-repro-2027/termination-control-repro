# Dataset format previews

**For inspection only. These files are not the full training or evaluation data.**

Each of the 28 JSONL files contains exactly the first 10 records of its
corresponding full dataset file, in the original order. Records are copied
without changing their contents. This is a format preview, not a random or
representative sample, and it must not be used to reproduce paper results.

The 10-row OBR manifest is also only a format example. Its references need not
match the first 10 records of the other files, so these previews do not form
a complete, consistent input for reconstruction or analysis.

Full data: [Termination Control Data](https://huggingface.co/datasets/tcr-repro-2027/termination-control-data).

```text
datasets/
  README.md
  PREVIEW_ONLY.json
  cleanv2/
    train_supportclean_keep8.jsonl
    eval_supportclean_keep8.jsonl
    swift_train_supportclean_keep8.jsonl
  raw/
    train_keep4{,_a,_ae}.jsonl
    swift_train_keep4{,_a,_ae}.jsonl
  controlled/
    train_{obr,obr_p15,obr_p10,obr_p5,isc_a,isc_e,isc_ae,benign_input,generic_noise}.jsonl
    swift_train_<same nine suffixes>.jsonl
    obr_pair_manifest.jsonl
```

Brace notation abbreviates ordinary filenames. `PREVIEW_ONLY.json` records
the row count and SHA256 of every preview file.

Record files contain `text`, `entities_str`, and `output`, a list of
`{source, target, relation, description}` objects. Evaluation records also
contain `key` and a corpus-category `source`. SWIFT files contain a `messages`
list with a user prompt and an assistant JSON target. The OBR manifest records
replacement locations and matching diagnostics.

For experiments, follow the [full-data setup instructions](../README.md#3-setup).
Download the full repository into `full_data/datasets/` and set `DATA_ROOT`
to that directory. Its `cleanv2/`, `raw/`, and `controlled/` directories must
be direct children of `DATA_ROOT`. This keeps the small previews available for
browsing while the experiment drivers use the complete inputs.

The code license does not relicense third-party document text. Consult the
full dataset card for the applicable data terms.
