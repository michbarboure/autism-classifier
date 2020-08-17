### An Automated Machine Learning Pipeline for Predicting Autism in Adults using EEG Data

## Abstract

The current diagnosis of autism spectrum disorder (ASD) relies solely on behavioural observations, leaving room for subjectivity and misdiagnosis. Studies have begun to address this issue by predicting ASD from brain imaging data using machine learning. However, few have focused their efforts in adults. This study aimed to determine whether electroencephalography (EEG) data could be used to train machine learning models that could accurately classify ASD individuals from a cohort of Dutch adults performing two visual tasks. 

We used a highly-automated preprocessing and feature engineering pipeline to limit biases in the methodology, then used grid search to select the highest performing models from a collection of five well-established learning algorithms. Of the two visual tasks, the boundary detection (BD) task resulted in the higher classification accuracy. By modifying the learning algorithms to include an intermediate regression step, our study was able to achieve higher mean classification performance than using traditional classification. It also provided a variable tolerance for sensitivity and specificity by tweaking the threshold parameter -- making this technique relevant for clinical applications. These enhancements were possible because both ASD diagnosis and Autism Quotient-Short scores were available as target features in the dataset. 

Tree-based learning algorithms obtained the highest performance, with the decision tree achieving a highest mean accuracy of 83\%. The most robust model was a random forest ensemble with 7 features and an area under the ROC curve of 0.887. 

Overall, this study provides strong evidence that ASD can be predicted with considerable accuracy in adults using EEG data. The paper also details a flexible and reproducible pipeline that can be used in future work for developing usable models.

---
