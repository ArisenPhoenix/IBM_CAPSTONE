# # if not already computed,
# # compute the sample weights to be used as input to the train routine so that
# # it takes into account the class imbalance present in this dataset
# # w_train = compute_sample_weight('balanced', y_train)
#
# # import the Decision Tree Classifier Model from Snap ML
# from snapml import DecisionTreeClassifier
#
# # Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
# # to use the GPU, set the use_gpu parameter to True
# # snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)
#
# # to set the number of CPU threads used at training time, set the n_jobs parameter
# # for reproducible output across multiple function calls, set random_state to a given integer value
# snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
#
# # train a Decision Tree Classifier model using Snap ML
# t0 = time.time()
# snapml_dt.fit(X_train, y_train, sample_weight=w_train)
# snapml_time = time.time()-t0
# print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))