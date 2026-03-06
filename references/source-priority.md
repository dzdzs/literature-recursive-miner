# Source Priority Rules

## Mandatory BibTeX rule

1. Query DBLP with paper title.
2. Choose the best DBLP match (prefer conference entry when equivalent).
3. Copy BibTeX from DBLP record endpoint.
4. If DBLP has no acceptable match, output `bibtex_status=missing_dblp`.

Never do the following:
- Never copy BibTeX from paper PDFs.
- Never copy BibTeX from random website snippets.
- Never generate or guess BibTeX fields.

## Paper URL priority

1. arXiv HTTPS PDF (`https://arxiv.org/pdf/<id>.pdf`)
2. OpenReview PDF/landing page
3. Other stable OA PDF or landing URL

## Time priority

1. arXiv ID month/year if available
2. Official venue publication date

## Citation expansion order

For each accepted paper:
1. Backward references (paper cites)
2. Forward citations (papers citing it)

Stop when:
- no new paper appears, or
- max paper cap is reached.
