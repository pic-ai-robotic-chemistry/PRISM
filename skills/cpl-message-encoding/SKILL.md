---
name: cpl-message-encoding
description: Encode and decode red-phase CPL target messages using colour-based Morse and base-4 CPL digits. Plan pixel layouts with measured CPL values or model-predicted peaks.
---

# CPL Message Encoding

Encode English letters and spaces in eight-pixel rows. Each pixel carries a Morse symbol through PL colour and a base-4 digit through CPL. See the [encoding and thermal readout rules](references/protocol.md).

## Design and encode

Run from the skill directory. Install the dependencies listed in [cof-inverse-design](../cof-inverse-design/SKILL.md). This workflow uses that skill's `assets/best.ckpt` to obtain red-phase parameters. It classifies the signed peak of greatest absolute magnitude in each predicted spectrum.

```bash
python scripts/encode_cpl_message.py --message "HELLO" --inverse-design --output-dir outputs/message
```

Use `--measurements data/cpl_cd_peaks.csv` to supply any subset of measured configurations. `design_targets.json` lists the requested pairs and any unresolved targets.

## Encode with measured data

The CSV requires `color`, `gray`, `thickness`, `stretch`, `is_red`, `peak_cd_intensity`, and `peak_wavelength`. Enter the signed CPL at the specified readout wavelength in `peak_cd_intensity` (mdeg) and the wavelength in `peak_wavelength` (nm). Use `is_red=True` for red phase. Enter PVA thickness in µm. The script copies the numeric thickness into the output field `thickness_nm`.

The standalone library builder requires all twelve colour and digit pairs.

```bash
python scripts/create_config_library.py --dataset data/cpl_cd_peaks.csv --output-dir outputs/library
python scripts/encode_cpl_message.py --message "HELLO" --library outputs/library/config_library.json --output-dir outputs/message
```

## Decode

For measured pixels, enter `observables.colour_symbol` and `observables.peak_cd_intensity`, then set `observables.source` to `measurement`.

```bash
python scripts/decode_cpl_message.py outputs/message/encoded_message.json --expected "HELLO"
```
