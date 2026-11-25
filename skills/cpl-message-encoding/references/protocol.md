# CPL Pixel Encoding Protocol

## Optical channels

Each pixel carries a photoluminescence (PL) colour and a signed CPL signal. Blue-emitting COF B represents a Morse dot, green-emitting COF G a dash, and red-emitting COF R a terminator or space. Read the CPL signal in mdeg at the specified wavelength and assign its base-4 digit using the table below.

| Digit | CPL at the readout wavelength (mdeg) | B | G | R |
|---:|---:|---|---|---|
| 0 | < −5000 | D001 | D005 | D009 |
| 1 | [−5000, 0) | D002 | D006 | D010 |
| 2 | [0, 5000] | D003 | D007 | D011 |
| 3 | > 5000 | D004 | D008 | D012 |

## Row layout

An eight-pixel row consists of two four-pixel blocks. The PL colours carry the first and third text characters in Morse. The CPL digits carry the second and fourth text characters as four-digit base-4 ASCII, most significant digit first. Interleave the decoded characters to recover the row text.

To encode `CPL` followed by a space, use colour blocks `G B G B | B G B B` for `C` and `L`, and CPL blocks `1100 | 0200` for `P` and the space.

## Thermal readout

Read the decoy message with PDA in the blue phase and the target message after heating converts it to the red phase. The transition changes the CPL interval for selected pixels. Each pixel keeps its PL colour and Morse symbol.
