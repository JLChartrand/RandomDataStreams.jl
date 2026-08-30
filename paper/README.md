# JOSS paper

This branch exists only to carry the JOSS submission. It was created from
`master` and is **never merged back**, which is the arrangement JOSS documents:

> Your paper (`paper.md` and BibTeX files, plus any figures) must be hosted in a
> Git-based repository together with your software. The paper may be in a
> short-lived branch which is never merged with the default, although if you do
> this, make sure this branch is *created* from the default so that it also
> includes the source code of your submission.

## Before submitting

**Recreate this branch from the then-current `master`.** It was branched from
`master` at a point where the PCG, Philox and Threefry work was still on the
`Philox` branch, so the source code alongside this paper is not yet the code the
paper describes. Once that work is merged, re-create `paper` from `master` and
replay these files onto it.

Working material — the state-of-the-field dossier with its evidence, the
editorial notes, the WSC paper — lives in the private companion repository
`fbastin/randomdatastreams-papers`. Only the finished paper belongs here.

## Conventions

Any measured figure quoted in `paper.md` carries the commit it was measured at,
as an HTML comment immediately above it.
