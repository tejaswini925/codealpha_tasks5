# ============================================================
# TASK 2: Exploratory Data Analysis (EDA)
# Dataset: Titanic
# Tools: Python, Pandas, Matplotlib, Seaborn
# Internship: CodeAlpha Data Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import io
warnings.filterwarnings('ignore')

# ── Load Dataset ─────────────────────────────────────────────
# You can replace this with: df = pd.read_csv('your_file.csv')

titanic_csv = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W",female,27,0,2,347742,11.1333,,S
10,1,2,"Nisson, Mrs. Alma",female,14,1,0,237736,30.0708,,C
11,1,3,"Sandstrom, Miss. Marguerite",female,4,1,1,PP 9549,16.7,G6,S
12,1,1,"Bonnell, Miss. Elizabeth",female,58,0,0,113783,26.55,C103,S
13,0,3,"Saundercock, Mr. William",male,20,0,0,A/5. 2151,8.05,,S
14,0,3,"Andersson, Mr. Anders",male,39,1,5,347082,31.275,,S
15,0,3,"Vestrom, Miss. Hulda",female,14,0,0,350406,7.8542,,S
16,1,2,"Hewlett, Mrs. Mary",female,55,0,0,248706,16,,S
17,0,3,"Rice, Master. Eugene",male,2,4,1,382652,29.125,,Q
18,1,2,"Williams, Mr. Charles",male,,0,0,244373,13,,S
19,0,3,"Vander Planke, Mrs.",female,31,1,0,345763,18,,S
20,1,3,"Masselmani, Mrs. Fatima",female,,0,0,2649,7.225,,C
21,0,2,"Fynney, Mr. Joseph",male,35,0,0,239865,26,,S
22,1,2,"Beesley, Mr. Lawrence",male,34,0,0,248698,13,D56,S
23,1,3,"McGowan, Miss. Anna",female,15,0,0,330923,8.0292,,Q
24,1,1,"Sloper, Mr. William",male,28,0,0,113788,35.5,A6,S
25,0,3,"Palsson, Miss. Torborg",female,8,3,1,349909,21.075,,S
26,1,3,"Asplund, Mrs. Carl",female,38,1,5,347077,31.3875,,S
27,0,3,"Emir, Mr. Farred",male,,0,0,2631,7.225,,C
28,0,1,"Fortune, Mr. Charles",male,19,3,2,19950,263,C23 C25 C27,S
29,1,3,"O'Dwyer, Miss. Ellen",female,,0,0,330959,7.8792,,Q
30,0,3,"Todoroff, Mr. Lalio",male,,0,0,349216,7.8958,,S
31,0,1,"Uruchurtu, Mr. Manuel",male,40,0,0,PC 17601,27.7208,,C
32,1,1,"Spencer, Mrs. William",female,,1,0,PC 17569,146.5208,B78,C
33,1,3,"Glynn, Miss. Mary",female,,0,0,335677,7.75,,Q
34,0,2,"Wheadon, Mr. Edward",male,66,0,0,C.A. 24579,10.5,,S
35,0,1,"Meyer, Mr. Edgar",male,28,1,0,PC 17604,82.1708,,C
36,0,1,"Holverson, Mr. Alexander",male,42,1,0,113789,52,,S
37,1,3,"Mamee, Mr. Hanna",male,,0,0,2677,7.2292,,C
38,0,3,"Cann, Mr. Ernest",male,21,0,0,A./5. 2152,8.05,,S
39,0,3,"Vander Planke, Miss. Augusta",female,18,2,0,345764,18,,S
40,1,3,"Nicola-Yarred, Miss. Jamila",female,14,1,0,2651,11.2417,,C
41,0,3,"Ahlin, Mrs. Johan",female,40,1,0,7546,9.475,,S
42,0,2,"Turpin, Mrs. William",female,27,1,0,11668,21,,S
43,0,3,"Kraeff, Mr. Theodor",male,,0,0,349253,7.8958,,C
44,1,2,"Laroche, Miss. Simonne",female,3,1,2,SC/Paris 2123,41.5792,,C
45,1,3,"Devaney, Miss. Margaret",female,19,0,0,330958,7.8792,,Q
46,0,3,"Rogers, Mr. William",male,,0,0,S.C./A.4. 23567,8.05,,S
47,0,3,"Lennon, Mr. Denis",male,,1,0,370371,15.5,,Q
48,1,3,"O'Driscoll, Miss. Bridget",female,,0,0,14311,7.75,,Q
49,0,3,"Samaan, Mr. Youssef",male,,2,0,2662,21.6792,,C
50,0,3,"Arnold-Franchi, Mrs. Josef",female,18,1,0,349237,17.8,,S
51,0,3,"Panula, Master. Juha",male,7,4,1,3101295,39.6875,,S
52,0,3,"Nosworthy, Mr. Richard",male,21,0,0,A/4. 39886,7.8,,S
53,1,1,"Harper, Mrs. Henry",female,49,1,0,PC 17572,76.7292,D33,C
54,1,2,"Faunthorpe, Mrs. Lizzie",female,29,1,0,2926,26,,S
55,0,1,"Ostby, Mr. Engelhart",male,65,0,1,113509,61.9792,B30,C
56,1,1,"Woolner, Mr. Hugh",male,,0,0,19947,35.5,C52,S
57,1,2,"Rugg, Miss. Emily",female,21,0,0,C.A. 31026,10.5,,S
58,0,3,"Novel, Mr. Mansouer",male,28,0,0,2697,7.2292,,C
59,1,2,"West, Miss. Constance",female,5,1,2,C.A. 34651,27.75,,S
60,0,3,"Goodwin, Master. William",male,11,5,2,CA 2144,46.9,,S
61,0,3,"Sirayanian, Mr. Orsen",male,22,0,0,2669,7.2292,,C
62,1,1,"Icard, Miss. Amelie",female,38,0,0,113572,80,B28,S
63,0,1,"Harris, Mr. Henry",male,45,1,0,36973,83.475,C83,S
64,0,3,"Skoog, Master. Harald",male,4,3,2,347088,27.9,,S
65,0,1,"Stewart, Mr. Albert",male,,0,0,PC 17605,27.7208,,C
66,1,3,"Moubarek, Master. Gerios",male,,1,2,2661,15.2458,,C
67,1,2,"Nye, Mrs. Elizabeth",female,29,0,0,C.A. 29395,10.5,F33,S
68,0,3,"Crease, Mr. Ernest",male,19,0,0,S.P. 3464,8.1583,,S
69,1,3,"Andersson, Miss. Erna",female,17,4,2,3101281,7.925,,S
70,0,3,"Kink, Mr. Vincenz",male,26,2,0,315151,8.6625,,S
71,0,2,"Jenkin, Mr. Stephen",male,32,0,0,C.A. 33111,10.5,,S
72,0,3,"Goodwin, Miss. Lillian",female,16,5,2,CA 2144,46.9,,S
73,0,2,"Hood, Mr. Ambrose",male,21,0,0,S.O.C. 14879,73.5,,S
74,0,3,"Chronopoulos, Mr. Apostolos",male,26,1,0,2680,14.4542,,C
75,1,3,"Bing, Mr. Lee",male,32,0,0,1601,56.4958,,S
76,0,3,"Moen, Mr. Sigurd",male,25,0,0,348123,7.65,F G73,S
77,0,3,"Staneff, Mr. Ivan",male,,0,0,349208,7.8958,,S
78,0,3,"Moutal, Mr. Rahamin",male,,0,0,374746,8.05,,S
79,1,2,"Caldwell, Master. Alden",male,0.83,0,2,248738,29,,S
80,1,3,"Dowdell, Miss. Elizabeth",female,30,0,0,364516,12.475,,S
81,0,3,"Waelens, Mr. Achille",male,22,0,0,345767,9,,S
82,1,3,"Sheerlinck, Mr. Jan",male,29,0,0,345779,9.5,,S
83,1,3,"McDermott, Miss. Delia",female,,0,0,330932,7.7875,,Q
84,0,1,"Carrau, Mr. Francisco",male,28,0,0,113059,47.1,,S
85,1,2,"Ilett, Miss. Bertha",female,17,0,0,SO/C 14885,10.5,,S
86,1,3,"Backstrom, Mrs. Karl",female,33,3,0,3101278,15.85,,S
87,0,3,"Ford, Mr. William",male,16,1,3,W./C. 6608,34.375,,S
88,0,3,"Slocovski, Mr. Selman",male,,0,0,SOTON/OQ 392086,8.05,,S
89,1,1,"Fortune, Miss. Mabel",female,23,3,2,19950,263,C23 C25 C27,S
90,0,3,"Celotti, Mr. Francesco",male,24,0,0,343275,8.05,,S
91,0,3,"Christmann, Mr. Emil",male,29,0,0,343276,8.05,,S
92,0,3,"Andreasson, Mr. Paul",male,20,0,0,347466,7.8542,,S
93,0,1,"Chaffee, Mr. Herbert",male,46,1,0,W.E.P. 5734,61.175,E31,S
94,0,3,"Dean, Mr. Bertram",male,26,1,2,C.A. 2315,20.575,,S
95,0,3,"Coxon, Mr. Daniel",male,59,0,0,364500,7.25,,S
96,0,3,"Shorney, Mr. Charles",male,,0,0,374910,8.05,,S
97,0,1,"Goldschmidt, Mr. George",male,71,0,0,PC 17754,34.6542,A5,C
98,1,1,"Greenfield, Mrs. Leo",female,45,0,1,PC 17759,63.3583,D35 E24,C
99,1,2,"Doling, Mrs. John",female,34,0,1,231919,23,,S
100,0,2,"Kantor, Mr. Sinai",male,34,1,0,244367,26,,S"""

df = pd.read_csv(io.StringIO(titanic_csv))

# ============================================================
# STEP 1: Data Structure
# ============================================================
print("=" * 55)
print("STEP 1: DATA STRUCTURE")
print("=" * 55)
print(f"Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ============================================================
# STEP 2: Missing Values
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: MISSING VALUES")
print("=" * 55)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ============================================================
# STEP 3: Descriptive Statistics
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: DESCRIPTIVE STATISTICS")
print("=" * 55)
print(df.describe())

# ============================================================
# STEP 4: Meaningful Questions & Findings
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: KEY QUESTIONS & FINDINGS")
print("=" * 55)

survival_rate = df['Survived'].mean() * 100
print(f"Q1. Overall survival rate       : {survival_rate:.1f}%")

by_sex = df.groupby('Sex')['Survived'].mean() * 100
print(f"\nQ2. Survival by Gender:\n{by_sex.to_string()}")

by_class = df.groupby('Pclass')['Survived'].mean() * 100
print(f"\nQ3. Survival by Passenger Class:\n{by_class.to_string()}")

avg_fare = df.groupby('Pclass')['Fare'].mean()
print(f"\nQ4. Average Fare by Class:\n{avg_fare.to_string()}")

print(f"\nQ5. Average Age : {df['Age'].mean():.1f} years")
print(f"    Min Age     : {df['Age'].min()} years")
print(f"    Max Age     : {df['Age'].max()} years")

# ============================================================
# STEP 5: Anomalies & Outliers
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: ANOMALIES & OUTLIERS")
print("=" * 55)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[df['Fare'] > Q3 + 1.5 * IQR]
print(f"Fare outliers detected : {len(outliers)} passengers")
print(f"Max fare               : £{df['Fare'].max():.2f}")
print(f"Min fare               : £{df['Fare'].min():.2f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
BG    = '#0f1117'
PANEL = '#1a1d2e'
C     = ['#4fc3f7','#f06292','#81c784','#ffb74d','#ce93d8']

fig = plt.figure(figsize=(18, 14), facecolor=BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
plt.suptitle('Titanic Dataset — Exploratory Data Analysis',
             fontsize=18, fontweight='bold', color='white', y=0.99)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.xaxis.label.set_color('#ccc')
    ax.yaxis.label.set_color('#ccc')
    ax.title.set_color('white')

# Plot 1: Survival Count
ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
survival_counts = df['Survived'].value_counts().sort_index()
ax1.bar(['Died', 'Survived'], survival_counts.values, color=[C[1], C[0]], width=0.5)
ax1.set_title('Survival Count', fontweight='bold')
for i, v in enumerate(survival_counts.values):
    ax1.text(i, v + 0.5, str(v), ha='center', color='white', fontweight='bold', fontsize=10)

# Plot 2: Survival by Gender
ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
gender_surv = df.groupby('Sex')['Survived'].mean() * 100
ax2.bar(gender_surv.index, gender_surv.values, color=[C[0], C[1]], width=0.4)
ax2.set_title('Survival Rate by Gender (%)', fontweight='bold')
ax2.set_ylim(0, 100)
for i, (k, v) in enumerate(gender_surv.items()):
    ax2.text(i, v + 1, f'{v:.1f}%', ha='center', color='white', fontweight='bold')

# Plot 3: Survival by Class
ax3 = fig.add_subplot(gs[0, 2]); style_ax(ax3)
class_surv = df.groupby('Pclass')['Survived'].mean() * 100
ax3.bar([f'Class {c}' for c in class_surv.index], class_surv.values, color=C[:3], width=0.5)
ax3.set_title('Survival Rate by Class (%)', fontweight='bold')
ax3.set_ylim(0, 100)
for i, v in enumerate(class_surv.values):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', color='white', fontweight='bold')

# Plot 4: Age Distribution
ax4 = fig.add_subplot(gs[1, 0]); style_ax(ax4)
age_data = df['Age'].dropna()
ax4.hist(age_data, bins=15, color=C[2], edgecolor=BG, alpha=0.85)
ax4.axvline(age_data.mean(), color=C[3], linestyle='--', linewidth=2,
            label=f'Mean: {age_data.mean():.1f}')
ax4.set_title('Age Distribution', fontweight='bold')
ax4.legend(labelcolor='white', facecolor=PANEL, fontsize=8)

# Plot 5: Fare Distribution
ax5 = fig.add_subplot(gs[1, 1]); style_ax(ax5)
ax5.hist(df['Fare'], bins=20, color=C[4], edgecolor=BG, alpha=0.85)
ax5.axvline(df['Fare'].median(), color=C[3], linestyle='--', linewidth=2,
            label=f'Median: £{df["Fare"].median():.1f}')
ax5.set_title('Fare Distribution (£)', fontweight='bold')
ax5.legend(labelcolor='white', facecolor=PANEL, fontsize=8)

# Plot 6: Embarkation Pie
ax6 = fig.add_subplot(gs[1, 2]); style_ax(ax6)
emb = df['Embarked'].value_counts()
emb_labels = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
ax6.pie(emb.values, labels=[emb_labels.get(k, k) for k in emb.index],
        colors=C[:len(emb)], autopct='%1.1f%%',
        textprops={'color': 'white', 'fontsize': 8},
        wedgeprops={'edgecolor': BG, 'linewidth': 2})
ax6.set_title('Embarkation Port', fontweight='bold')

# Plot 7: Missing Values
ax7 = fig.add_subplot(gs[2, 0]); style_ax(ax7)
miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
miss_pct = miss_pct[miss_pct > 0]
ax7.barh(miss_pct.index, miss_pct.values, color=C[1])
ax7.set_title('Missing Values (%)', fontweight='bold')
ax7.set_xlabel('% Missing')
for i, v in enumerate(miss_pct.values):
    ax7.text(v + 0.3, i, f'{v:.1f}%', va='center', color='white', fontsize=8)

# Plot 8: Correlation Heatmap
ax8 = fig.add_subplot(gs[2, 1]); style_ax(ax8)
num_cols = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(num_cols, ax=ax8, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, linecolor=BG,
            annot_kws={'color': 'white', 'size': 8}, cbar=False)
ax8.set_title('Correlation Heatmap', fontweight='bold')
ax8.tick_params(colors='#ccc', labelsize=7)

# Plot 9: Age vs Fare Scatter
ax9 = fig.add_subplot(gs[2, 2]); style_ax(ax9)
died = df[df['Survived'] == 0]
surv = df[df['Survived'] == 1]
ax9.scatter(died['Age'], died['Fare'], color=C[1], alpha=0.6, s=25, label='Died')
ax9.scatter(surv['Age'], surv['Fare'], color=C[0], alpha=0.6, s=25, label='Survived')
ax9.set_title('Age vs Fare (by Survival)', fontweight='bold')
ax9.set_xlabel('Age')
ax9.set_ylabel('Fare (£)')
ax9.legend(labelcolor='white', facecolor=PANEL, fontsize=8)

plt.savefig('task2_eda_output.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nChart saved as: task2_eda_output.png")
print("\n✅ Task 2 - EDA Complete!")
