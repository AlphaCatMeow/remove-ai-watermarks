# Provenance fixtures

These files exercise real container, metadata, and pixel-provenance formats.
They are test inputs, not evaluation corpora.

## Adobe TrustMark

`adobe-trustmark-p.png` is Adobe's official `images/ghost_P.png` Variant P
example from [`adobe/trustmark`](https://github.com/adobe/trustmark) at commit
`0ed40cbe8188f664fd9cbbeacd969807de27440a`.

- Source: `https://raw.githubusercontent.com/adobe/trustmark/0ed40cbe8188f664fd9cbbeacd969807de27440a/images/ghost_P.png`
- SHA-256: `e58c5825ed7e5d9fb04710ea541b61bd55879cad65554c0d46260aa24b3d0755`
- Expected signal: TrustMark Variant P, schema 1
- License: MIT, reproduced in `../licenses/adobe-trustmark-MIT.txt`


## Ideogram C2PA attribution

`ideogram-c2pa.png` is a SYNTHETIC signed manifest, not a vendor download: a
64x64 flat PNG carrying a C2PA claim (`trainedAlgorithmicMedia`,
`claim_generator "Ideogram/3.0"`) signed by a throwaway test CA whose leaf
certificate subject is `O=Ideogram, Inc, CN=Ideogram, Inc`. It exercises the
issuer-to-platform ATTRIBUTION path end to end through the real reader; the
signer is deliberately untrusted (no trust anchors ship), which is also the
fixture's second assertion: attribution and signer trust are separate
dimensions.

- Expected signal: C2PA, platform `Ideogram`, `ai_from_metadata` True
- Expected signer: `untrusted` (test CA; attribution must not imply trust)
- Recipe (no keys are committed): create a CA and an ES256 leaf with
  `digitalSignature` key usage and subject `CN=Ideogram, Inc`, then
  `c2pa.Builder.from_json(...).sign_file(png, out, Signer.from_info(
  C2paSignerInfo(b"es256", leaf_pem + ca_pem, pkcs8_key, None)))`.
- No user upload and no Ideogram artifact contributes pixels or bytes.
