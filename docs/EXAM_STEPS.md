# Examination checklist

Use only the existing PE-CA2 repository. Work on application code and infrastructure now; import the genuine Colab model later.

1. **Dataset:** `D:\telegram\archive\housing.csv` matches `data/housing.csv`. Target is `median_house_value`; predictors are the other nine columns.
2. **Colab evidence:** save the executed notebook, runtime information, train/test split, and measured regression metrics. The metadata currently reports 200 trees, R² 0.8181, MAE 31486.8013, and RMSE 48823.4524; these are reported results, not verified local results.
3. **Application now:** install dependencies and run `pytest tests/test_api.py -v` with the project interpreter. Start Uvicorn. `/live` and `/docs` work without the model; `/health`, `/model-info`, and valid `/predict` return 503. Invalid input returns 422. Run the smoke test with `--allow-missing-model`.
4. **Docker now:** follow README.md to build and run the image and check real HTTP endpoints in the pending-model state. Capture the build and smoke-test output.
5. **Import later:** place `california_housing_rf_model.pkl` and `california_housing_rf_metadata.json` in `models/`. The saved object must be the full fitted pipeline. Restart Uvicorn and run `scripts/verify_model.py`, all pytest tests, and the smoke test without the allow-missing flag. Rebuild and rerun Docker and repeat the strict smoke test.
6. **Existing GitHub repository:** keep origin at `https://github.com/paarth293/PE-CA2.git`. Commit code now if ready; commit the model later, confirming its file is under GitHub's 100 MiB limit. Never commit passwords or tokens.
7. **CI completion:** before importing the model, `code-and-container` can pass while `real-model-validation` is skipped and the run summary records that it is pending. After importing and pushing the model, both jobs must run and pass. Save the successful final run URL and screenshots. Do not call a skipped model job successful verification.

Submission evidence: dataset link and schema; executed Colab notebook and metrics; genuine fitted model and metadata; API and test results; Docker predictions; repository URL; final Actions run with both jobs successful.
