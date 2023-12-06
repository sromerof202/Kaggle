import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
import json

# Load config
with open("config.json") as config_file:
    config = json.load(config_file)

# Load your data
data = pd.read_csv(config["url"])

# Identifying UTAUT constructs columns and role column
ordinal_cols = [col for col in data.columns if "Questions/Statements" in col]
role_column = "1.What is your role at the university?"

# Creating a composite score from the UTAUT constructs
label_encoder = LabelEncoder()
encoded_data = data[ordinal_cols].apply(lambda col: label_encoder.fit_transform(col))

# Calculating the mean score across UTAUT constructs for each respondent
data["Composite_UTAUT_Score"] = encoded_data.mean(axis=1)

# Grouping data based on roles at the university
groups = data.groupby(role_column)["Composite_UTAUT_Score"]

# Preparing data for ANOVA
students_scores = groups.get_group("Student")
faculty_scores = groups.get_group("Faculty")
staff_scores = groups.get_group("Staff")

# Conducting ANOVA
anova_result = f_oneway(students_scores, faculty_scores, staff_scores)

print(f"ANOVA p-value: {anova_result.pvalue}")
print(f"ANOVA F-statistic: {anova_result.statistic}")
