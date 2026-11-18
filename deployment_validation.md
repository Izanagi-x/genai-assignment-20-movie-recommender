# Live deployment validation

Checked on 18 November 2026 using Chrome, after Render reported the deployment Live.

- Repository: https://github.com/Izanagi-x/genai-assignment-20-movie-recommender
- App: https://genai-assignment-20-movie-recommender.onrender.com/
- Tested application commit: `a5ecca67e1398fb85577f538368e800d886c6710`
- Host: Render Free web service, Python 3.14.8, main branch.
- Render health check: `/_stcore/health`.
- The dropdown loaded all 4,803 movies and used year/ID labels to distinguish titles.

| Selected movie | Five displayed recommendations in order |
| --- | --- |
| Avatar | Aliens; Alien³; Moonraker; Alien; Silent Running |
| Toy Story | Toy Story 2; Toy Story 3; Small Soldiers; The 40 Year Old Virgin; Child's Play 2 |
| Fight Club | Dead Man's Shoes; The Machinist; The New Guy; Se7en; The Talented Mr. Ripley |

Every query returned five items without recommending itself. The displayed scores
matched the local model to three decimal places. Local checks independently
recomputed cosine similarities, checked rankings, invalid inputs and duplicate
titles, and exercised the same three queries with Streamlit AppTest.
The extracted ZIP also ran `main.py` offline successfully.

The live UI is a functional smoke test, not a user relevance study. Initial
startup on the free service took about a minute; later interactions were faster.
