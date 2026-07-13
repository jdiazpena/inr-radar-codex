# Synthetic Regularization Sweep Findings

Baseline `xy030_t030` is the best overall run in this sweep.

- Best full-domain density RMSE: `xy030_t030` at 2.6305e10 m^-3.
- Best full-domain combined gradient RMSE: `xy030_t030` at 3.2015e-03.
- Best interior density and gradient: `xy030_t030`.
- Best near-observation density and gradient: `xy030_t030`.
- High-gradient density RMSE is lowest for `data_only`, but high-gradient gradient RMSE is lowest for `xy030_t030`.

Answers to the comparison questions:

1. Data-only fits the sparse observations very strongly, but its derivative diagnostics and gradient errors are larger than the baseline, consistent with more oscillatory behavior between points.
2. Baseline `0.30/0.30` regularization gives the best all-around density and gradient behavior, suggesting it reduces artifacts without excessive smearing.
3. Stronger spatial regularization `0.70/0.30` does not improve gradient recovery here; it is worse than baseline in full-domain combined gradient RMSE.
4. Stronger spatial and temporal regularization `0.70/0.70` appears too strong for this case; it worsens full-domain density RMSE and does not improve gradients.
5. Best density reconstruction overall: `xy030_t030`, with the caveat that `data_only` is best for density RMSE inside the high-gradient region only.
6. Best gradient reconstruction: `xy030_t030` across all regions.
7. The best density and gradient run are the same except for the high-gradient density-only comparison.
