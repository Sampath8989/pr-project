# Can top-choice accuracy be increased?

A text ranker scored **7/13** on the previously inspected test set, compared with the BN's **5/13**. This was exploratory and is not a clean estimate of improvement.

For a broader check, five question-group folds cover all 91 four-agent source traces. The BN got **40/91** correct; the text ranker got **33/91**. The ranker therefore did not improve top-choice accuracy across groups. Its Brier score was 0.654 versus 0.696 for the BN; that is a different metric.

The deployed BN stays in place. A convincing accuracy increase needs more representative labeled traces, a method selected without looking at a new test set, and an untouched external evaluation.
