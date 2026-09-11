# ARTISTIC public source audit

- Public historical source: `https://github.com/ravinsingh166/Manufacturing-Model-Codes.git`
- Pinned upstream commit: `5af9e0345673fac557c479ee8f4a0727e442c1fa` (`update 11/22`)
- Retrieval: 2026-09-11T12:53:19.5144317+07:00; source subtree `NMC/Updated version` git-tree `e870c018acf8696424ae3ac9c36163f5bd3afb56`; repository git-tree `b4720e1b7cc1d7b51b47efc1443ad3bcafc5a011`.
- License: CC BY-NC-SA 4.0, as identified for ARTISTIC source material. Its usage constraints remain applicable.

This is public historical model code, not the restricted Zenodo 5956128 package. Zenodo remains an independently restricted artifact; its restricted status does not prohibit the pinned public GitHub source. The source README requires sequential slurry → drying → calendering execution and documents cluster-specific scripts. AIcoScientist replaces those scripts with a portable local/MPI/SLURM runner while preserving the source inputs and recording commands, versions, hashes, and outputs per run.

Before every prepare or execution, the runner fail-closes unless the checkout is clean, its `HEAD` is the pinned commit, and the `NMC/Updated version` git-tree equals the audited hash above.
