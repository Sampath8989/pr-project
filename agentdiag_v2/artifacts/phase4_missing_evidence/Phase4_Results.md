# Phase 4 — Missing-log inference

- **complete:** 5/13 correct (38.5%); Brier 0.689; ECE 0.146; log loss 1.238.
- **middle_A3_missing:** 5/13 correct (38.5%); Brier 0.699; ECE 0.097; log loss 1.262.
- **one_random_missing:** 6/13 correct (46.2%); Brier 0.711; ECE 0.177; log loss 1.295.
- **two_random_missing:** 4/13 correct (30.8%); Brier 0.729; ECE 0.099; log loss 1.333.

Missing evidence is marginalized by VE; it is not treated as a clean log. Every scenario uses the validation-fitted temperature. Masks are fixed by the recorded seed.
