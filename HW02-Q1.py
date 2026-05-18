
###########
### Q01 ###
###########
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
clothes = pd.read_excel("hw02.xlsx", sheet_name="clothes")

# Convert 'type' column to a categorical variable
clothes['type'] = clothes['type'].astype('category')
print(clothes['type'].describe())

# D. One-way ANOVA
model_aov = ols('price ~ C(type)', data = clothes).fit()
anova_results = anova_lm(model_aov, typ = 2)
print("ANOVA Results:\n", anova_results)

# 1. חישוב הממוצע הכללי של כל התצפיות (מחירי כל הבגדים יחד)
grand_mean = clothes['price'].mean()
# 2. חישוב הממוצע הפנימי של כל סוג חנות בנפרד
group_means = clothes.groupby('type')['price'].mean()
# 3. חישוב האמדנים: הפחתת הממוצע הכללי מהממוצע של כל חנות
tau_estimates = group_means - grand_mean
print("\n--- אמדני הסטייה השיטתית לכל חנות (Tau) ---")
print(tau_estimates)