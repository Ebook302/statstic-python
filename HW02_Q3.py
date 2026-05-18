###########
### Q03git ###
###########
import pandas as pd
df_rats = pd.read_csv("rats.csv")
#  סופרים כמה תצפיות יש לכל שילוב של 'poison' ו-'treat'
balancing_check = df_rats.groupby(['poison', 'treat']).size()
print("\n--- בדיקת איזון הניסוי (כמות תצפיות לכל שילוב) ---")
print(balancing_check)

# --- Q3 סעיף ג' - ניתוח שונות דו-כיווני (Two-Way ANOVA) ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot

# 1. בניית המודל והפקת טבלת ANOVA
model_rats = ols('time ~ C(poison) * C(treat)', data=df_rats).fit()
anova_table_rats = sm.stats.anova_lm(model_rats, typ=2)
print("\n--- טבלת ANOVA לשאלה 3 ---")
print(anova_table_rats)


# 2. יצירת גרף אינטראקציה (קריטי להבנת סעיף ג')
fig, ax = plt.subplots(figsize=(10, 6))
# x = הגורם על ציר ה-X, trace = הקווים השונים, response = המשתנה הנמדד
interaction_plot(x=df_rats['poison'], trace=df_rats['treat'], response=df_rats['time'],
                 ax=ax, markers=['D','^','o','s'], ms=10)

plt.title("Interaction Plot: Poison vs Treatment Effect on Survival Time")
plt.xlabel("Poison Type")
plt.ylabel("Mean Survival Time")
plt.legend(title="Treatment Type")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
# 1. חילוץ השאריות מהמודל של שאלה 3
residuals_rats = model_rats.resid
# 2. מבחן פורמלי לנורמליות - Shapiro-Wilk
shapiro_stat, shapiro_p = stats.shapiro(residuals_rats)

# 3. מבחן פורמלי לשוויון שונויות - Levene
# ב-ANOVA דו-כיווני, בודקים שוויון שונויות לפי השילוב של כל הקבוצות (התאים בטבלה)
levene_stat, levene_p = stats.levene(*[group["time"].values for name, group in df_rats.groupby(['poison', 'treat'])])

print("\n--- בדיקת הנחות מודל שאלה 3 (פורמלי) ---")
print(f"נורמליות (Shapiro-Wilk): P-value = {shapiro_p:.4f}")
print(f"שוויון שונויות (Levene): P-value = {levene_p:.4f}")


print("\n--- בדיקת הנחות מודל שאלה 3 (ויזואלי) ---")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# 1. חילוץ השאריות והערכים החזויים מהמודל
residuals = model_rats.resid
fitted = model_rats.fittedvalues
# 2. יצירת לוח לגרפים (1 על 2)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# --- גרף א': בדיקת נורמליות (Q-Q Plot) ---
stats.probplot(residuals, dist="norm", plot=ax[0])
ax[0].set_title('Normal Q-Q Plot')
ax[0].set_xlabel('Theoretical Quantiles')
ax[0].set_ylabel('Sample Quantiles')

# --- גרף ב': בדיקת שוויון שונויות (Residuals vs Fitted) ---
ax[1].scatter(fitted, residuals, color='blue', edgecolor='k', alpha=0.6)
ax[1].axhline(y=0, color='red', linestyle='--')
ax[1].set_title('Residuals vs Fitted (Homoscedasticity Check)')
ax[1].set_xlabel('Fitted Values')
ax[1].set_ylabel('Residuals')
plt.tight_layout()
plt.show()

# Q3- סעיף ה
import pandas as pd
df_rats = pd.read_csv("rats.csv")

# --- שלב 1: ההשערות הראשוניות (מתוך טבלת ה-ANOVA שכבר הרצנו) ---
# את התוצאות האלו את לוקחת מהטבלה של סעיף ג':
# Poison: PV < 0.05 -> יש הבדל מובהק בין הרעלים (מצדיק פוסט-הוק)
# Treat: PV < 0.05 -> יש הבדל מובהק בין הטיפולים (מצדיק פוסט-הוק)
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
def run_fdr_analysis(df, factor):
    # יצירת אובייקט השוואה
    comp = mc.MultiComparison(df['time'], df[factor])
    # ביצוע t-tests בין כל הזוגות עם תיקון FDR (בשיטת Benjamini-Hochberg)
    # שימי לב: השתמשתי ב-'fdr_bh' ישירות בתוך הפונקציה
    result_table, p_adjusted, _ = comp.allpairtest(stats.ttest_ind, method='fdr_bh')
    return result_table

# הרצה עבור רעלים
print("\n--- השוואות זוגיות: רעלים (Poison) עם תיקון FDR ---")
print(run_fdr_analysis(df_rats, 'poison'))
# הרצה עבור טיפולים
print("\n--- השוואות זוגיות: טיפולים (Treat) עם תיקון FDR ---")
print(run_fdr_analysis(df_rats, 'treat'))