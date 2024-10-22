- reviews_data.csv : all negative reviews - (review_text, hotel_name, deceptive_flag: 1 if deceptive)
- utils.py: a few helper functions for actions such as performing grid search and writing results
- classifiers.py: executing the grid search for the 4 classifiers, and training the models with the best hyperparameters
                - create_descriptors(): creates feature vectors for the text in the reviews
                - if we want to test more hyperparameters, add to: logreg_grid_search(), tree_grid_search(), rf_search()
                - kfold_no_features() can be used to test with cross-validation for feature selection (based on word/feature popularity), although the create_descriptors() has the option to do it too. 

-grid_search_result: the full results of the grid searches and specified best parameters

-model_eval: the results after training the model with the best parameters and testing it on a test set
