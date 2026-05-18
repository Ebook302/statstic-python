###########
### Q02 ###
###########
### A
print("Q2 :\n")
import matplotlib.pyplot as plt
import seaborn as sns
invest = pd.read_excel("hw02.xlsx", sheet_name = "invest")

# Create a 2x2 subplot for each strategy
strategies = invest['channel'].unique()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot histograms for each strategy
for i, strategy in enumerate(strategies):
    sns.histplot(invest[invest['channel'] == strategy]['return'], bins = 15, kde = True, ax = axes[i])
    axes[i].set_title(f'Distribution of Returns - Strategy {strategy}')
    axes[i].set_xlabel('Annual Return')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

### B
# Rename the 'return' column to avoid syntax errors
invest_renamed = invest.rename(columns={'return': 'returns'})

# Fit the ANOVA model with the renamed column
model = ols('returns ~ C(channel)', data = invest_renamed).fit()
anova_table = anova_lm(model, typ = 2)

print(f"anova_table is {anova_table}")


#Q2 סעיף ב
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# טעינת הנתונים
invest = pd.read_excel("hw02.xlsx", sheet_name="invest")
# שינוי שם העמודה 'return' ל-'returns' כי 'return' היא מילה שמורה בפייתון ויכולה לעשות בעיות
invest_renamed = invest.rename(columns={'return': 'returns'})
# אמידת המודל ויצירת טבלת ניתוח שונות (ANOVA)
model = ols('returns ~ C(channel)', data=invest_renamed).fit()
anova_table = anova_lm(model, typ=2)
print("--- טבלת ניתוח שונות (ANOVA) - סעיף ב' ---")
print(anova_table)

#Q2 - 3
print("Q2 סעיף ג :")
import pandas as pd
import statsmodels.stats.multicomp as mc
import scipy.stats as stats
# יצירת האובייקט
comp = mc.MultiComparison(invest_renamed['returns'], invest_renamed['channel'])
# נבקש תיקון בונפרוני (method='b') רק כדי שהפונקציה תסכים לרוץ
raw_table = comp.allpairtest(stats.ttest_ind, method='b')
print("=====================================================")
print(" טבלת השוואות זוגיות (קחי רק את עמודת ה- pval !) ")
print("=====================================================")
print(raw_table[0])

# --- Q2 סעיף ג' - מבחן TUKEY HSD ---
from statsmodels.stats.libqsturng import qsturng
import numpy as np
# 1. הכנת נתוני העזר לחישובים הידניים
mse = model.mse_resid  # שונות השאריות מטבלת ה-ANOVA
df_resid = model.df_resid  # דרגות חופש של השאריות
k = invest_renamed['channel'].nunique()  # מספר הקבוצות (4)
n_per_group = invest_renamed.groupby('channel').size().iloc[0]  # גודל מדגם (8)
# 2. מציאת ערך Q קריטי (לפי אלפא 0.05) - לשימוש בהשוואה ידנית
q_critical = qsturng(1 - 0.05, k, df_resid)
# 3. הרצת המבחן וחילוץ הטבלה המלאה כולל p-values
tukey_res = comp.tukeyhsd()
tukey_summary = tukey_res.summary().data
tukey_df = pd.DataFrame(tukey_summary[1:], columns=tukey_summary[0])

print(f"\n--- פרמטרים למבחן Tukey ---")
print(f"MSE: {mse:.4f}")
print(f"ערך Q קריטי (מהטבלה): {q_critical:.4f}")
print("\n--- טבלת Tukey HSD סופית (כולל p-adj) ---")

# עמודת p-adj היא ה-p-value המתוקן
print(tukey_df[['group1', 'group2', 'meandiff', 'p-adj', 'reject']])

print("\n--- פירוט חישוב Q_observed לכל זוג (עבור סעיף ג') ---")
means = invest_renamed.groupby('channel')['returns'].mean()
channels = sorted(invest_renamed['channel'].unique())
for i in range(len(channels)):
    for j in range(i + 1, len(channels)):
        c1, c2 = channels[i], channels[j]
        diff = np.abs(means[c1] - means[c2])
        # הנוסחה ל-Q מחושב
        q_obs = diff / np.sqrt(mse / n_per_group)
        print(f"השוואה {c1}-{c2}: Q_obs = {q_obs:.4f}")


print("Q2 סעיף ד :")
# Q2 - סעיף ד': מבחן Levene לשוויון שונויות
from scipy.stats import levene

# 1. פירוק הנתונים לקבוצות לפי אפיק השקעה
groups = [group["returns"].values for name, group in invest_renamed.groupby("channel")]

# 2. הרצת המבחן
levene_stat, levene_p = levene(*groups)
print("--- סעיף ד': מבחן Levene לשוויון שונויות ---")
print(f"סטטיסטי המבחן (W): {levene_stat:.4f}")
print(f"p-value: {levene_p:.4f}")


# --- Q2 סעיף ה' - ויזואליזציה ללא דריסת כותרות ---
import matplotlib.pyplot as plt
import seaborn as sns

# שימוש ב-constrained_layout=True פותר את בעיית הכותרות הדורסות
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.flatten()

strategies = invest_renamed['channel'].unique()

for i, strategy in enumerate(strategies):
    sns.histplot(invest_renamed[invest_renamed['channel'] == strategy]['returns'], bins=10, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution - Channel {strategy}', fontsize=14, pad=10)
    axes[i].set_xlabel('Annual Return')
    axes[i].set_ylabel('Frequency')

plt.show()

# 1. המבחן הפורמלי: Shapiro-Wilk על שאריות המודל
# הערה: בודקים את השאריות כי זה משקף את הנורמליות של כל המודל יחד
shapiro_stat, shapiro_p = stats.shapiro(model.resid)
print("\n--- סעיף ה': מבחן פורמלי לנורמליות (Shapiro-Wilk) ---")
print(f"Statistic: {shapiro_stat:.4f}, P-value: {shapiro_p:.4f}")

