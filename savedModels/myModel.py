import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


dataf = pd.read_csv('malaria_clinical_data.csv')
#input("View Dataset")
#print(dataf)
#input("Press enter for replacing empty cells with the median")

dataf["temperature"].fillna(dataf["temperature"].median(axis=0), inplace=True)
dataf["parasite_density"].fillna(dataf["parasite_density"].median(axis=0), inplace=True)
dataf["wbc_count"].fillna(dataf["wbc_count"].median(axis=0), inplace=True)
dataf["rbc_count"].fillna(dataf["rbc_count"].median(axis=0), inplace=True)
dataf["hb_level"].fillna(dataf["hb_level"].median(axis=0), inplace=True)
dataf["hematocrit"].fillna(dataf["hematocrit"].median(axis=0), inplace=True)
dataf["mean_cell_volume"].fillna(dataf["mean_cell_volume"].median(axis=0), inplace=True)
dataf["mean_corp_hb"].fillna(dataf["mean_corp_hb"].median(axis=0), inplace=True)
dataf["mean_cell_hb_conc"].fillna(dataf["mean_cell_hb_conc"].median(axis=0), inplace=True)
dataf["platelet_count"].fillna(dataf["platelet_count"].median(axis=0), inplace=True)
dataf["platelet_distr_width"].fillna(dataf["platelet_distr_width"].median(axis=0), inplace=True)
dataf["mean_platelet_vl"].fillna(dataf["mean_platelet_vl"].median(axis=0), inplace=True)
dataf["neutrophils_percent"].fillna(dataf["neutrophils_percent"].median(axis=0), inplace=True)
dataf["lymphocytes_percent"].fillna(dataf["lymphocytes_percent"].median(axis=0), inplace=True)
dataf["mixed_cells_percent"].fillna(dataf["mixed_cells_percent"].median(axis=0), inplace=True)
dataf["neutrophils_count"].fillna(dataf["neutrophils_count"].median(axis=0), inplace=True)
dataf["lymphocytes_count"].fillna(dataf["lymphocytes_count"].median(axis=0), inplace=True)
dataf["mixed_cells_count"].fillna(dataf["mixed_cells_count"].median(axis=0), inplace=True)
# dataf["RBC_dist_width_Percent"].fillna(dataf["RBC_dist_width_Percent"].median(axis=0),inplace=True)
#print (dataf.to_string())

#input("Press enter to highlight the duplicates")

#print(dataf.duplicated())

#input("Press enter for Removing  the duplicate")
dataf.drop_duplicates(inplace = True)

X = dataf.drop(columns=['SampleID', 'consent_given', 'location', 'Enrollment_Year', 'bednet', 'fever_symptom', 'Suspected_Organism', 'Suspected_infection', 'RDT', 'Blood_culture', 'Urine_culture', 'Taq_man_PCR', 'Microscopy', 'Laboratory_Results', 'Clinical_Diagnosis', 'RBC_dist_width_Percent' ])
y = dataf["Clinical_Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Decision Tree, Logistic Regression, Support Vector Machine  and Random Forest Classifiers
#input("Press enter for create classifiers")
Decision_tree_model = DecisionTreeClassifier()
Logistic_regression_Model = LogisticRegression(solver='lbfgs', max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)
# Train the models using the training sets
#input("Press enter to train the model")

Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
print("hello einstein")
RF_model.fit(X_train, y_train)

# Predict the response for test dataset
#input("Press enter to test the model")
print(X_test)
DT_Prediction = Decision_tree_model.predict(X_test)
LR_Prediction = Logistic_regression_Model.predict(X_test)
SVM_Prediction = SVM_model.predict(X_test)
RF_Prediction = RF_model.predict(X_test)
# Calculation of Model Accuracy
#input("Press enter to calculate the accuracy")
DT_acc = accuracy_score(y_test, DT_Prediction)
LR_acc = accuracy_score(y_test, LR_Prediction)
SVM_acc = accuracy_score(y_test, SVM_Prediction)
RF_acc = accuracy_score(y_test, RF_Prediction)
print("hello world");
#input("Press enter to print the accuracy data...")

print("Decistion Tree accuracy =", DT_acc * 100, "%")
print("Logistic Regression accuracy =", LR_acc * 100, "%")
print("Suport Vector Machine accuracy =", SVM_acc * 100, "%")
print("Random Forest accuracy =", RF_acc * 100, "%")


#input("Press enter to persist the model...")
prediction = Decision_tree_model.predict(X_test)

joblib.dump(Decision_tree_model, 'malaria_detector2.joblib')

persistedModel = joblib.load('malaria_detector2.joblib')

temperature = float(input("Enter temperature :"))
parasite_density = int(input("Enter parasite_density: "))
wbc_count = float(input("Enter wbc_count:"))
rbc_count = float(input("Enter rbc_count: "))
hb_level = float(input("Enter hb_level :"))
hematocrit = float(input("Enter hematocrit: "))
mean_cell_volume = int(input("Enter mean_cell_volume :"))
mean_corp_hb = float(input("Enter mean_corp_hb: "))
mean_cell_hb_conc = float(input("Enter mean_cell_hb_conc :"))
platelet_count = float(input("Enter platelet_count: "))
platelet_distr_width = float(input("Enter platelet_distr_width :"))
mean_platelet_vl = float(input("Enter mean_platelet_vl: "))
neutrophils_percent = float(input("Enter neutrophils_percent :"))
lymphocytes_percent = float(input("Enter lymphocytes_percent: "))
mixed_cells_percent = float(input("Enter mixed_cells_percent :"))
neutrophils_count = float(input("Enter neutrophils_count: "))
lymphocytes_count = float(input("Enter lymphocytes_count :"))
mixed_cells_count= float (input ("Enter mixed_cells_count: "))
# RBC_dist_width_Percent=int (input ("Enter Mid Exam Marks Marks :"))


model = joblib.load('malaria_detector2.joblib')
predictions = model.predict([[temperature, parasite_density, wbc_count, rbc_count, hb_level, hematocrit, mean_cell_volume, mean_corp_hb, mean_cell_hb_conc, platelet_count, platelet_distr_width, mean_platelet_vl, neutrophils_percent, lymphocytes_percent, mixed_cells_percent, neutrophils_count, lymphocytes_count, mixed_cells_count]])
print("The Grade you will obtain is:", predictions)
