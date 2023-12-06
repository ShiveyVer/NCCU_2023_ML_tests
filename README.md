## TEST1
- **Preprocessing**
    none
        
- **Prediction**

    *RandomForestClassifier*
    
    training_data.csv (training)
    testing_data.csv (for prediction)
    ⇒ predict.py ⇒ predicted_results.csv
## TEST2
- **Preprocessing**
    1. *replacing missing value with mean*
        
        training_data.csv ⇒ impute.py ⇒ data_filled_mean.csv
        
    2. *resample data with SMOTE*
        
        data_filled_mean.csv ⇒ resample.csv ⇒ data_resampled.csv
        
- **Prediction**
    
    *scaling then use tf.keras.models.Sequential for training/prediction*
    
    data_resampled.csv (training)
    testing_data.csv (for prediction)
    ⇒ predict.py ⇒ predicted_results.csv