import streamlit as st
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import tensorflow as tf
from aif360.algorithms.inprocessing import AdversarialDebiasing
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from scipy.stats import chi2_contingency


# Function to calculate bias metrics
# def calculate_bias_metrics(dataset_orig, target_column, feature):
#     # Initialize a dictionary to store bias metrics
#     bias_metrics = {}

#         # Create a BinaryLabelDataset
#     dataset = StandardDataset(dataset_orig, label_name=target_column, favorable_classes=[1],
#                                   protected_attribute_names=[feature], privileged_classes=[[1]])

#         # Calculate fairness metrics using BinaryLabelDatasetMetric
#     metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{feature: 0}],
#                                           privileged_groups=[{feature: 1}])

#     classified_metric = ClassificationMetric(dataset,
#                                                 dataset,
#                                                 unprivileged_groups=[{feature: 0}],
#                                                 privileged_groups=[{feature: 1}])

#         # Store bias metrics in the dictionary
#     bias_metrics[feature] = {
#             "Disparate Impact": metric.disparate_impact(),
#             "Statistical Parity Difference": metric.statistical_parity_difference(),
#             "Equal Opportunity Difference": classified_metric.equal_opportunity_difference()
#         }

#     return bias_metrics

def calculate_bias_metrics(df, target_column, feature_columns):
    # Initialize a dictionary to store bias metrics
    bias_metrics = {}

    # Initialize lists to store fairness metrics for visualization
    disparate_impact_values = []
    statistical_parity_difference_values = []
    
    # Initialize a list to store protected attributes with disparate impact greater than one
    protected_attributes_with_disparate_impact = []

    # Detect bias for each feature
    for protected_attribute in feature_columns:
        # Create a BinaryLabelDataset
        dataset = StandardDataset(df, label_name=target_column, favorable_classes=[1],
                                  protected_attribute_names=[protected_attribute], privileged_classes=[[1]])

        # Calculate fairness metrics using BinaryLabelDatasetMetric
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protected_attribute: 0}],
                                          privileged_groups=[{protected_attribute: 1}])

        # Store bias metrics in the dictionary
        bias_metrics[protected_attribute] = {
            "Protected Attribute": protected_attribute,
            "Disparate Impact": metric.disparate_impact(),
            "Statistical Parity Difference": metric.statistical_parity_difference(),
        }
        
        st.subheader(f"Metrics for feature '{protected_attribute}':")
        
        for metric_name, metric_value in bias_metrics[protected_attribute].items():
            st.write(f"{metric_name}: {metric_value}")

        # Append fairness metric values to lists for visualization
        disparate_impact_values.append(metric.disparate_impact())
        statistical_parity_difference_values.append(metric.statistical_parity_difference())

        # Check if disparate impact is greater than one and add to the list
        if metric.disparate_impact() > 1.6:
            protected_attributes_with_disparate_impact.append(protected_attribute)

    # Print protected attributes with disparate impact greater than one
    st.write(f"Protected Attributes: {protected_attributes_with_disparate_impact}")
    # print(protected_attributes_with_disparate_impact)
    
    st.title('Fairness Metrics Visualization')

    # Visualize fairness metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Disparate Impact
    ax1.bar(feature_columns, disparate_impact_values, color='blue')
    ax1.set_ylabel('Disparate Impact')
    ax1.set_title('Disparate Impact for Each Feature')

    # Plot Statistical Parity Difference
    ax2.bar(feature_columns, statistical_parity_difference_values, color='orange')
    ax2.set_ylabel('Statistical Parity Difference')
    ax2.set_title('Statistical Parity Difference for Each Feature')

    plt.tight_layout()

    # Show the plot using Streamlit
    st.pyplot(fig)

    # Return the protected attributes array
    return protected_attributes_with_disparate_impact

# Function to perform preprocessing
def preprocess_data(df, target_column, protected_attribute):
    bias_metrics = {}
    
    # Convert the Pandas DataFrame to a BinaryLabelDataset
    binary_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[target_column],
        protected_attribute_names=[protected_attribute]
    )

    # Apply reweighing preprocessing algorithm
    preprocessor = Reweighing(unprivileged_groups=[{protected_attribute: 0}],
                               privileged_groups=[{protected_attribute: 1}])
    preprocessed_dataset = preprocessor.fit_transform(binary_dataset)

    # Convert the preprocessed dataset back to a Pandas DataFrame
    preprocessed_df = preprocessed_dataset.convert_to_dataframe()
    
    metric = BinaryLabelDatasetMetric(preprocessed_dataset, unprivileged_groups=[{protected_attribute: 0}],
                                          privileged_groups=[{protected_attribute: 1}])

    classified_metric = ClassificationMetric(preprocessed_dataset,
                                                preprocessed_dataset,
                                                unprivileged_groups=[{protected_attribute: 0}],
                                                privileged_groups=[{protected_attribute: 1}])

    # Store bias metrics in the dictionary
    bias_metrics[protected_attribute] = {
        "Disparate Impact": metric.disparate_impact(),
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Equal Opportunity Difference": classified_metric.equal_opportunity_difference()
    }
    
    # Display bias metrics
    for metric_name, metric_value in bias_metrics[protected_attribute].items():
        st.write(f"{metric_name}: {metric_value}")
        
    # st.title('Fairness Metrics Visualization')

    # # Visualize fairness metrics
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # # Plot Disparate Impact
    # ax1.bar(feature_columns, disparate_impact_values, color='blue')
    # ax1.set_ylabel('Disparate Impact')
    # ax1.set_title('Disparate Impact for Each Feature')

    # # Plot Statistical Parity Difference
    # ax2.bar(feature_columns, statistical_parity_difference_values, color='orange')
    # ax2.set_ylabel('Statistical Parity Difference')
    # ax2.set_title('Statistical Parity Difference for Each Feature')

    # plt.tight_layout()

    # # Show the plot using Streamlit
    # st.pyplot(fig)

    return preprocessed_df

def training_data(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [5, 7, 9],
        'reg_lambda': [0.1, 0.5, 1]
    }

    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=7, reg_lambda=0.1)
    xgb_clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = xgb_clf.predict(X_test)

    # Get the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate additional evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.write("### Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    
    st.write("### Confusion Matrix:")
    st.pyplot(conf_matrix)
    
    return conf_matrix

def training_debiasing(df, target, features):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    
    protected_attribute = 'sex'

    # Convert the Pandas DataFrame to a BinaryLabelDataset
    binary_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[target],
        protected_attribute_names=[protected_attribute]
    )
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define individual classifiers
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=7, reg_lambda=0.1)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create a Voting Classifier
    voting_clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf)], voting='hard')

    # Fit the ensemble model
    voting_clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred_ensemble = voting_clf.predict(X_test)

    # Get the confusion matrix
    conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_ensemble, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Ensemble (XGBoost + Random Forest)')
    plt.show()

    # Calculate additional evaluation metrics
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    precision_ensemble = precision_score(y_test, y_pred_ensemble)
    recall_ensemble = recall_score(y_test, y_pred_ensemble)
    f1_ensemble = f1_score(y_test, y_pred_ensemble)
    
    st.write(f"Accuracy: {accuracy_ensemble}")
    st.write(f"Precision: {precision_ensemble}")
    st.write(f"Recall: {recall_ensemble}")
    st.write(f"F1 Score: {f1_ensemble}")

    # Apply AdversarialDebiasing in-processing algorithm
    with tf.compat.v1.variable_scope('debiased_classifier', reuse=tf.compat.v1.AUTO_REUSE):
        adversarial_debiasing = AdversarialDebiasing(unprivileged_groups=[{protected_attribute: 0}],
                                                    privileged_groups=[{protected_attribute: 1}],
                                                    scope_name='debiased_classifier',
                                                    debias=True, sess=tf.compat.v1.Session())

    adversarial_debiasing.fit(binary_dataset)

    # Get predictions on the original dataset
    predictions = adversarial_debiasing.predict(binary_dataset)

    # Create a copy of the original dataset with predicted labels
    debiasing_dataset = binary_dataset.copy(deepcopy=True)
    debiasing_dataset.labels = predictions.labels

    # Evaluate the debiased dataset for fairness metrics
    metric = BinaryLabelDatasetMetric(debiasing_dataset,
                                    unprivileged_groups=[{protected_attribute: 0}],
                                    privileged_groups=[{protected_attribute: 1}])

    classified_metric = ClassificationMetric(debiasing_dataset,
                                            debiasing_dataset,
                                            unprivileged_groups=[{protected_attribute: 0}],
                                            privileged_groups=[{protected_attribute: 1}])
    
    st.write(f"Disparate Impact: {metric.disparate_impact()}")
    st.write(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
    st.write(f"Equal Opportunity Difference: {classified_metric.equal_opportunity_difference()}")

    return conf_matrix_ensemble

        

# Streamlit app
def main():
    # Streamlit UI
    st.title("Bias Metrics Calculator")

# Upload File
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
    # Load the dataset
        dataset_orig = pd.read_csv(uploaded_file)
        protected_attribute = 'sex'
        
        # Display the gender bias
        plt.figure(figsize=(15, 5))
        sns.countplot(data=dataset_orig, x=dataset_orig[protected_attribute], hue=dataset_orig['target'])
        plt.xlabel('Gender')
        plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
        plt.title('Distribution of Samples by Gender and Target', fontsize=15)
        plt.grid(alpha=0.4)
        st.pyplot(plt)
        
        # Display the gender chest pain
        plt.figure(figsize=(15,5))
        sns.countplot(data=dataset_orig, x=dataset_orig['cp'], hue=dataset_orig['target'])
        plt.xlabel('Chest Pain')
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        plt.title('Distribution of Samples by Chest Pain and Target', fontsize=15)
        plt.grid(alpha=0.4)
        st.pyplot(plt)
        
        # Display correlation heatmap
        st.write("### Correlation Heatmap")
        corr_matrix = dataset_orig.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax = sns.heatmap(corr_matrix,
                         annot=True,
                         linewidths=0.5,
                         fmt=".2f",
                         cmap="YlGnBu")

        # Display the plot in Streamlit
        st.pyplot(fig)

    # Extract target column and feature columns
        target_column = st.selectbox("Select Target Column", dataset_orig.columns)
        feature_columns = st.multiselect("Select Feature Columns", dataset_orig.columns)
        
        protected_attributes = calculate_bias_metrics(dataset_orig, target_column, feature_columns)

        # Protecting each feature one at a time and calculating bias metrics
        # for feature in feature_columns:
        #     st.subheader(f"Metrics for feature '{feature}':")
        #     bias_metrics = calculate_bias_metrics(dataset_orig, target_column, feature)
            
        #     # Display bias metrics
        #     for metric_name, metric_value in bias_metrics[feature].items():
        #         st.write(f"{metric_name}: {metric_value}")
        
        preprocessed_df = None
        conf_matrix = None
        
        if st.button("Start Debiasing", key='start_preprocessing'):
            with st.spinner("Dibiasing in progress..."):
                # Perform preprocessing
                preprocessed_df = preprocess_data(dataset_orig, target_column, 'sex')
                time.sleep(3)
        
        # st.write("### Training Process")
        # if st.button("Start Training", key='training'):
        #     with st.spinner("Training in progress..."):
        #         if preprocessed_df is not None:
        #             conf_matrix = training_data(preprocessed_df, target_column, feature_columns)
        #             time.sleep(3)

        #         if conf_matrix is not None:
        #             st.write("### Training Completed")
        #         else:
        #             st.write("### Training Failed")
                
                
        # st.write("### Training and Debiasing Progress")
        # if st.button("Start Training", key='training_debiasing'):
        #     with st.spinner("Training in progress..."):
        #         conf_matrix_ensemble = training_debiasing(preprocessed_df, target_column, feature_columns)
        #         time.sleep(3)
                
            # if conf_matrix_ensemble is not None:
            #     st.write("### Training & Debiasing Completed")
            #     st.pyplot(conf_matrix_ensemble)
            # else:
            #     st.write("### Training & Debiasing Failed")
                
        if preprocessed_df is not None:
            st.write("### Debiasing Completed")

if __name__ == "__main__":
    main()
        
