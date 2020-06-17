# PH125.9x-movielens
Repository for project 1 of Edx Harvard Data Science Capstone course (PH125.9x) - MovieLens

## Important files in this repository
1. `get_data.R` was provided as part of the project setup in our courseware. This file retrieves the data we need to analyse and splits it into a "training" set named `edx` and a validation set to be used at the very end of our analysis to measure the resulting accuracy of our work.
2. `analysis.R` leverages `get_data.R`, saves the data to disk and then proceeds to fully analyse the data. The last portion of this script applies the final model to the validation set for a final measure of accuracy.
3. `report.Rmd` is an R-markdown file that generates a final report.
4. `report.pdf` is the final report generated for this project. *NB - this report does not utilise the full 10M record dataset for sanity's sake.*