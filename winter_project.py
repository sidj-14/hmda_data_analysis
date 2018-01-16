import pandas as pd 
import numpy as np
import scipy.stats as stats
## Importing dataset
data_2007 = pd.read_csv("hmda_lar_2007.csv",delimiter=",")
data_2016 = pd.read_csv("hmda_lar_2016.csv",delimiter=",")

# data_2007.replace(r'\s+', np.nan, inplace=True,regex=True)
# data_2016.replace(r'\s+', np.nan, inplace=True,regex=True)

# column_names = data_2007.columns
# missing_values = pd.DataFrame(column_names, columns=["Column_Name"])
# missing_values["missing_values_2007"] = -1
# missing_values["missing_values_2016"] = -1
# n_columns_full = len(column_names)

# for i in list(range(n_columns_full)):
# 	data_2007.loc[i,"missing_values_2007"] = data_2007[column_names[i]].isnull().sum()
# 	data_2016.loc[i,"missing_values_2016"] = data_2016[column_names[i]].isnull().sum()

# print(missing_values)

# ## DROPPED COLUMN INFORMATION
# #Property_Type_Name = One-to-four family dwelling (other than manufactured housing)
# #Owner_Occupancy_Name = Owner-occupied as a principal dwelling
# #lien_status_name = Secured by a first lien
drop_columns = ["preapproval_name","property_type_name","hoepa_status_name","owner_occupancy_name","lien_status_name","agency_name",
"purchaser_type_name","tract_to_msamd_income","rate_spread","state_name","msamd_name","applicant_race_name_5","applicant_race_name_4",
"applicant_race_name_3","applicant_race_name_2","sequence_number","respondent_id","denial_reason_name_3","denial_reason_name_2",
"denial_reason_name_1","county_name","co_applicant_sex_name","co_applicant_race_name_5","co_applicant_race_name_4","co_applicant_race_name_3",
"co_applicant_race_name_2","co_applicant_race_name_1","co_applicant_ethnicity_name","census_tract_number","application_date_indicator",
"agency_abbr","edit_status_name"]
data_2007 = data_2007.drop(drop_columns,axis=1)
data_2016 = data_2016.drop(drop_columns,axis=1)


def convert_data_set(data,year_ind):
	# Income Categorization
	data["income_cat"] = pd.cut(data["applicant_income_000s"],bins=[0,25,50,75,100,150,200,250,10000],labels=[1,2,3,4,5,6,7,8])
	data["applicant_income_000s"] = data["applicant_income_000s"]*1000
	#Minority Population
	data["min_population"] = pd.cut(data["minority_population"],bins=[0,2.5,5,10,25,50,100],labels=[1,2,3,4,5,6])
	# Median Family Income
	data["med_income"] = pd.cut(data["hud_median_family_income"],bins=[0,25000,50000,75000,100000],labels=[1,2,3,4])
	# Income more than median flag
	data["income_median_flag"] = 0
	data.loc[(data["applicant_income_000s"]) >= data["hud_median_family_income"],"income_median_flag"] = 1
	# State Abbr 
	data["state"] = 0
	data.loc[data["state_abbr"]=='CA',"state"] = 1
	data.loc[data["state_abbr"]=='NY',"state"] = 2
	# Loan Type Name
	data["loan_type"] = 0
	data.loc[data["loan_type_name"] == "Conventional","loan_type"] = 1
	data.loc[data["loan_type_name"] == "FHA-insured","loan_type"] = 2
	data.loc[data["loan_type_name"] == "VA-guaranteed","loan_type"] = 2
	data.loc[data["loan_type_name"] == "FSA/RHS-guaranteed","loan_type"] = 2
	# Loan Purpose Name
	data["loan_purpose"] = 0
	data.loc[data["loan_purpose_name"]== "Home purchase","loan_purpose"] = 1
	data.loc[data["loan_purpose_name"]== "Refinancing","loan_purpose"] = 2
	# Applicant Sex Name
	data["gender"] = 0
	data.loc[data["applicant_sex_name"]=="Male","gender"] = 1
	# Applicant Race 
	data["race"] = 0
	data.loc[data["applicant_race_name_1"] == "White", "race"] = 1
	data.loc[data["applicant_race_name_1"] == "Black or African American", "race"] = 2
	data.loc[data["applicant_race_name_1"] == "Native Hawaiian or Other Pacific Islander", "race"] = 3
	data.loc[data["applicant_race_name_1"] == "Information not provided by applicant in mail, Internet, or telephone application", "race"] = 4
	data.loc[data["applicant_race_name_1"] == "Asian", "race"] = 5
	data.loc[data["applicant_race_name_1"] == "American Indian or Alaska Native", "race"] = 6
	data.loc[data["applicant_race_name_1"] == "Not applicable", "race"] = 7
	# Year
	data["year"] = year_ind
	# Action Taken on the loan
	data["loan_action"] = 0
	data.loc[data["action_taken_name"]=="Loan originated","loan_action"] = 1
	# "Loan approved but not accepted" is 0.
	data.rename(columns={"applicant_income_000s":"income","number_of_1_to_4_family_units":"1-4_Homes","number_of_owner_occupied_units":"owner_occupied"},inplace=True)
	data = data.drop(["hud_median_family_income","state_abbr","loan_type_name","loan_purpose_name",
		"as_of_year","applicant_sex_name","applicant_race_name_1","applicant_ethnicity_name","action_taken_name"],axis = 1)
	return data


new_data_2007 = convert_data_set(data_2007,1)
new_data_2007 = new_data_2007[~((new_data_2007.race == 3) | (new_data_2007.race == 4) | (new_data_2007.race == 6) | (new_data_2007.race == 7))]
new_data_2016 = convert_data_set(data_2016,2)
new_data_2016 = new_data_2016[~((new_data_2016.race == 3) | (new_data_2016.race == 4) | (new_data_2016.race == 6) | (new_data_2016.race == 7))]


# CHecking missing values in the formatted dataset
column_names = new_data_2007.columns
missing_values = pd.DataFrame(column_names, columns=["Column_Name"])
missing_values["missing_values_2007"] = -1
missing_values["missing_values_2016"] = -1
n_columns_full = len(column_names)

for i in list(range(n_columns_full)):
	missing_values.loc[i,"missing_values_2007"] = new_data_2007.iloc[:,i].isnull().sum()
	missing_values.loc[i,"missing_values_2016"] = new_data_2016.iloc[:,i].isnull().sum()

print(missing_values)

print(new_data_2007.shape)
print(new_data_2016.shape)
new_data_2007.dropna(axis=0,how='any',inplace=True)
new_data_2007.reset_index(drop=True,inplace=True)
new_data_2016.dropna(axis=0,how='any',inplace=True)
new_data_2016.reset_index(drop=True,inplace=True)
print("\n")
print(new_data_2007.shape)
print(new_data_2016.shape)


for i in list(range(n_columns_full)):
	missing_values.loc[i,"missing_values_2007"] = new_data_2007.iloc[:,i].isnull().sum()
	missing_values.loc[i,"missing_values_2016"] = new_data_2016.iloc[:,i].isnull().sum()

print(missing_values)

new_data_2007.to_csv("data_2007_hmda.csv", index=False, header = True)
new_data_2016.to_csv("data_2016_hmda.csv", index=False, header = True)









