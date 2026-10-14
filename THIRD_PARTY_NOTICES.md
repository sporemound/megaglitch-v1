# Third-party notices

The existing root GPLv3 license is preserved for this repository. It does not replace the original notices or terms of separately credited media, references and bundled dependencies.

## Application source

The `app/` source snapshot comes from **sporemound/megaglitch**, inspected parent commit `ff8b53c9376c271c1b7ce8ca95f38065f3073603`. Original source comments, release requirements, test corpus and documentation are retained. Runtime binaries and developer caches are excluded from Git.

## Red Corona example poster

`preset/source-poster.png` and `preset/source-poster.svg` use the actual **USAF** photograph of a C-119J of the 6593d with its recovery gear extended during a training parachute recovery in 1959. The supplied reference is *CORONA Star Catchers*, NRO Center for the Study of National Reconnaissance, June 2012, PDF page 36 / printed page 25. The editable SVG retains the photo credit and the exact stream address, `https://kick.com/berry-punch`.

The style reference is the **Uniflora aesthetic kit**, credited to Uniflora contributors, source <https://github.com/sporemound/uniflora-aesthetic-kit>, commit `92acf469da1a832c0bad510472dd2a31a832acd8`, nocturnal-optics branch. Reference artwork/documentation is **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>); reference code is **Unlicense**. The archival-photo composition, midnight-blue panels, silver lettering and teal accents are a new interpretation for this poster. Those separate scopes and the change notice are preserved in `preset/poster-provenance.json`.

The poster discusses Tim Glister's *Red Corona* and the historical CORONA program. It contains no reproduced book text or book cover.

## Soundtrack

`preset/soundtrack.wav` is the original synthetic 60-second stereo cue created for this example with MegaGlitchFM synthesis voices. No recording or reference-video audio was sampled. Musical and mastering details are recorded in `preset/soundtrack-notes.json`.

## Portable-runtime dependencies

Python and dependency license notices must remain in the generated runtime. A supplied static FFmpeg archive must retain its license, README/build details and documentation in the generated package. Optional custom OpenCV/CUDA inputs retain their manifests, licenses and provenance. No third-party runtime DLL is checked into this Git repository.
