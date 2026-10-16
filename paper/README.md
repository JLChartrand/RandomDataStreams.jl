# JOSS paper

This branch exists only to carry the JOSS submission. It was created from
`master` and is **never merged back**, which is the arrangement JOSS documents:

> Your paper (`paper.md` and BibTeX files, plus any figures) must be hosted in a
> Git-based repository together with your software. The paper may be in a
> short-lived branch which is never merged with the default, although if you do
> this, make sure this branch is *created* from the default so that it also
> includes the source code of your submission.

## Before submitting

**Bring this branch up to the then-current `master`** if `master` has moved on
since, so that the source code alongside the paper is the code it describes.
The branch adds nothing outside `paper/`, so `git rebase master` replays it
cleanly.

Working material — the state-of-the-field dossier with its evidence, the
editorial notes, the WSC paper — lives in the private companion repository
`fbastin/randomdatastreams-papers`. Only the finished paper belongs here.

## Conventions

Any measured figure quoted in `paper.md` carries the commit it was measured at,
as an HTML comment immediately above it.
