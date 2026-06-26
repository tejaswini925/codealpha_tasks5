# ============================================================
# TASK 3: Data Visualization
# Dataset: Titanic
# Tools: Python, Pandas, Matplotlib, Seaborn
# Internship: CodeAlpha Data Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# ── Color Palette ─────────────────────────────────────────────
BG     = '#080c14'
PANEL  = '#111827'
BORDER = '#1f2d40'
TEAL   = '#00d4aa'
CORAL  = '#ff6b6b'
GOLD   = '#ffd166'
PURPLE = '#a78bfa'
BLUE   = '#60a5fa'
WHITE  = '#f0f4f8'
MUTED  = '#94a3b8'

def style(ax, title=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)

# ── Figure Setup ──────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=BG)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4,
                       left=0.05, right=0.97, top=0.91, bottom=0.06)

fig.text(0.5, 0.965, 'TITANIC SURVIVAL ANALYSIS', fontsize=22,
         fontweight='bold', color=WHITE, ha='center', fontfamily='monospace')
fig.text(0.5, 0.945, 'A data story about who survived — and why',
         fontsize=11, color=MUTED, ha='center')
fig.add_artist(plt.Line2D([0.05, 0.95], [0.935, 0.935],
               color=TEAL, linewidth=1.5, transform=fig.transFigure))

# ── Chart 1: KPI Cards ────────────────────────────────────────
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor(BG)
for sp in ax_kpi.spines.values(): sp.set_visible(False)
ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

kpis = [
    ('100',                           'Total Passengers', TEAL),
    (f"{df['Survived'].sum()}",       'Survived',         TEAL),
    (f"{100-df['Survived'].sum()}",   'Perished',         CORAL),
    (f"{df['Survived'].mean()*100:.0f}%", 'Survival Rate', GOLD),
    (f"£{df['Fare'].mean():.0f}",     'Avg Ticket Fare',  PURPLE),
    (f"{df['Age'].mean():.0f} yrs",   'Average Age',      BLUE),
]
for i, (val, label, col) in enumerate(kpis):
    x = 0.08 + i * 0.155
    rect = FancyBboxPatch((x - 0.065, 0.05), 0.125, 0.88,
                          boxstyle="round,pad=0.02", transform=ax_kpi.transAxes,
                          facecolor=PANEL, edgecolor=col, linewidth=1.5)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 0.62, val, ha='center', va='center', color=col,
                fontsize=18, fontweight='bold', transform=ax_kpi.transAxes)
    ax_kpi.text(x, 0.25, label, ha='center', va='center', color=MUTED,
                fontsize=8, transform=ax_kpi.transAxes)

# ── Chart 2: Survival by Class ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0]); style(ax2, '① Survival by Passenger Class')
class_data = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
x = np.arange(3); w = 0.35
ax2.bar(x - w/2, class_data[0], w, color=CORAL, label='Died',     zorder=3)
ax2.bar(x + w/2, class_data[1], w, color=TEAL,  label='Survived', zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], color=MUTED)
ax2.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

# ── Chart 3: Gender Survival Rate ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 1]); style(ax3, '② Gender Survival Rate')
gender = df.groupby('Sex')['Survived'].mean() * 100
colors_g = [TEAL if g == 'female' else CORAL for g in gender.index]
bars = ax3.barh(gender.index, gender.values, color=colors_g, height=0.4)
ax3.set_xlim(0, 100)
ax3.xaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
ax3.set_axisbelow(True)
for bar, v in zip(bars, gender.values):
    ax3.text(v + 1, bar.get_y() + bar.get_height() / 2,
             f'{v:.1f}%', va='center', color=WHITE, fontsize=9, fontweight='bold')
ax3.tick_params(axis='y', colors=WHITE)

# ── Chart 4: Age Group Stacked Bar ────────────────────────────
ax4 = fig.add_subplot(gs[1, 2]); style(ax4, '③ Age Group — Survived vs Died')
age_surv = df.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)
age_labels = [str(x) for x in age_surv.index]
died_vals = age_surv[0].values if 0 in age_surv.columns else np.zeros(len(age_surv))
surv_vals = age_surv[1].values if 1 in age_surv.columns else np.zeros(len(age_surv))
x_pos = np.arange(len(age_labels))
ax4.bar(x_pos, died_vals, color=CORAL, label='Died',     zorder=3)
ax4.bar(x_pos, surv_vals, bottom=died_vals, color=TEAL, label='Survived', zorder=3)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(age_labels, color=MUTED, fontsize=7, rotation=15)
ax4.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
ax4.set_axisbelow(True)
ax4.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=8)

# ── Chart 5: Embarkation Donut ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 3]); style(ax5, '④ Embarkation Port')
emb = df['Embarked'].value_counts()
emb_labels = {'S': 'Southampton\n(S)', 'C': 'Cherbourg\n(C)', 'Q': 'Queenstown\n(Q)'}
wedges, texts, autotexts = ax5.pie(
    emb.values, labels=[emb_labels.get(k, k) for k in emb.index],
    colors=[BLUE, GOLD, PURPLE][:len(emb)], autopct='%1.0f%%',
    pctdistance=0.75, startangle=90,
    textprops={'color': WHITE, 'fontsize': 8},
    wedgeprops={'edgecolor': BG, 'linewidth': 2.5, 'width': 0.55}
)
for at in autotexts:
    at.set_color(BG); at.set_fontweight('bold')

# ── Chart 6: Fare Box Plot ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0]); style(ax6, '⑤ Fare Distribution by Class')
data_by_class = [df[df['Pclass'] == c]['Fare'].dropna().values for c in [1, 2, 3]]
bp = ax6.boxplot(data_by_class, patch_artist=True,
                 medianprops={'color': BG, 'linewidth': 2},
                 whiskerprops={'color': MUTED},
                 capprops={'color': MUTED},
                 flierprops={'marker': 'o', 'markersize': 4,
                             'markerfacecolor': CORAL, 'alpha': 0.6})
for patch, col in zip(bp['boxes'], [BLUE, GOLD, PURPLE]):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax6.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], color=MUTED)
ax6.set_ylabel('Fare (£)')
ax6.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
ax6.set_axisbelow(True)

# ── Chart 7: Age vs Fare Scatter ──────────────────────────────
ax7 = fig.add_subplot(gs[2, 1:3]); style(ax7, '⑥ Age vs Fare — Each Dot is a Passenger')
died_df = df[df['Survived'] == 0].dropna(subset=['Age', 'Fare'])
surv_df = df[df['Survived'] == 1].dropna(subset=['Age', 'Fare'])
ax7.scatter(died_df['Age'], died_df['Fare'], color=CORAL, alpha=0.6, s=40,
            label='Died', edgecolors='none', zorder=3)
ax7.scatter(surv_df['Age'], surv_df['Fare'], color=TEAL, alpha=0.7, s=40,
            label='Survived', edgecolors='none', zorder=4)
ax7.set_xlabel('Age (years)')
ax7.set_ylabel('Ticket Fare (£)')
ax7.xaxis.grid(True, color=BORDER, linestyle='--', alpha=0.4)
ax7.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.4)
ax7.set_axisbelow(True)
ax7.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
max_fare_row = df.loc[df['Fare'].idxmax()]
ax7.annotate(f'Highest fare\n£{max_fare_row["Fare"]:.0f}',
             xy=(30, max_fare_row['Fare']), xytext=(45, 240),
             color=GOLD, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL,
                       edgecolor=GOLD, linewidth=1))

# ── Chart 8: Heatmap — Class x Gender ────────────────────────
ax8 = fig.add_subplot(gs[2, 3]); style(ax8, '⑦ Survival Rate: Class × Gender')
pivot = df.pivot_table(values='Survived', index='Sex',
                       columns='Pclass', aggfunc='mean') * 100
sns.heatmap(pivot, ax=ax8, annot=True, fmt='.0f', cmap='RdYlGn',
            linewidths=1.5, linecolor=BG,
            annot_kws={'color': 'black', 'size': 11, 'weight': 'bold'},
            cbar=False, vmin=0, vmax=100)
ax8.set_xticklabels(['1st', '2nd', '3rd'], color=MUTED)
ax8.set_yticklabels(['Female', 'Male'], color=MUTED, rotation=0)
ax8.set_xlabel('Passenger Class')
ax8.set_ylabel('')

# Footer
fig.text(0.5, 0.025,
         'Data Visualization  |  Task 3  |  Titanic Dataset (n=100)  |  Tools: Python · Matplotlib · Seaborn',
         ha='center', color=MUTED, fontsize=8, fontstyle='italic')

plt.savefig('task3_visualization_output.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Chart saved as: task3_visualization_output.png")
print("\n✅ Task 3 - Data Visualization Complete!")
