def testing_models(X, y, models, balance):
    """
    Trains and evaluates a list of classification models under different data preprocessing scenarios.

    The process includes:
    - Evaluating the models with two configurations: with and without outliers.
    - Testing three types of scaling: no scaling, standardization, and normalization.
    - Applying two class balancing techniques: SMOTE and SMOTEENN, or no balancing.
    - Splitting the data into training and test sets for each configuration.
    - Preprocessing the data for each scenario, removing outliers if applicable, and applying the chosen scaling method.
    - Applying class balancing techniques (SMOTE or SMOTEENN) only on the training set when specified.
    - Training each model on the preprocessed data.
    - Evaluating the performance of the models on the test set using the following metrics:
      - Accuracy
      - Precision (focus on minimizing false negatives)
      - Recall
      - F1-score
    - Storing the results of each model along with the outlier, scaling, and balancing configuration used.
    - Storing the confusion matrices for each model and scenario.

    Returns:
    - A DataFrame containing the performance metrics of each model under each scenario (outliers, scaling, and balancing).
    - A dictionary storing the confusion matrices generated for each model and configuration.
    """

    # Prepare results DataFrame
    results = []
    conf = {}
    i = 0

    # Iterate through different scenarios
    for outliers in ['With Outliers', 'Without Outliers']:
        for scaling in ['No Scaling', 'Standardization', 'Normalization']:
            # Prepare data
            X_subset = X.copy()

            if outliers == 'Without Outliers':
                X_subset = remove_outliers(X_subset)   
                y_subset = y[X_subset.index]
            else:
                y_subset = y

            X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

            if scaling == 'Standardization':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            elif scaling == 'Normalization':
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            if balance == 1:
                sm=SMOTE(random_state=42)
                X_train_scaled,y_train=sm.fit_resample(X_train_scaled,y_train)

            elif balance == 2:
                smote_enn = SMOTEENN(random_state=42, sampling_strategy=0.9)
                X_train_scaled, y_train = smote_enn.fit_resample(X_train_scaled, y_train)

            # Train and evaluate models
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred )
                f1 = f1_score(y_test, y_pred)
                conf[i] = confusion_matrix(y_test, y_pred)
                i += 1

                results.append({
                    'Model': model_name,
                    'Outliers': outliers,
                    'Scaling': scaling,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1score': f1
                })

    results_df = pd.DataFrame(results)

    return results_df, conf
