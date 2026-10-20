# AI-assisted contributions

AI-assisted contributions are allowed, but the human contributor remains the author and is responsible for the
change.

## Required disclosure

Every pull request that used an AI tool must include:

```text
AI use
- Tool/model:
- Extent: planning / implementation / tests / documentation / review
- Human verification:
```

"AI-assisted" alone is not sufficient. State what the tool produced and what the contributor verified.

## Contributor responsibilities

The contributor must:

- understand and be able to explain the change without relying on the AI transcript;
- review and edit generated text and code before submission;
- run the relevant tests, and the GPU e2e when behavior changes, and report their actual output;
- remove prompt artifacts, speculative comments, and duplicated documentation.

Do not submit:

- private prompts, chat transcripts, credentials, internal repository names, hostnames, or unpublished baselines;
- AI-generated images, audio, or video; rendered simulation output from RoboSmith itself is fine;
- fabricated tests, success rates, benchmark results, citations, or claims;
- AI co-author trailers unless the repository history adopts such a convention.

The same review standard applies whether or not AI was used.
