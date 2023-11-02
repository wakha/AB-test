# ----------
# AB test
# ----------

"""Case Study"""

""" A company recently introduced a new bidding type, “average bidding”, as an alternative to 
its existing bidding type, called “maximum bidding”. One of our clients, non_real_bidding_company.com, 
has decided to test this new feature and wants to conduct an A/B test to understand if 
the new feature brings more average revenue than maximum bidding. """

# Importing packages
# --------------------
import pandas as pd
from scipy.stats import kstest, mannwhitneyu
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pprint import pprint

# Importing data
# --------------

control_df = pd.read_csv("./data/control_group.csv", delimiter=";")
test_df = pd.read_csv("./data/test_group.csv", delimiter=";")

df = pd.concat([control_df, test_df])
df.head()

# Pre-processing and cleaning of data
# -----------------------------------

"""Resetting index to dates"""
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
df = df.set_index("Date")  # setting date column as index

"""Checking duplicates"""
df[
    "Campaign Name"
].unique()  # Making sure that there are only two variants for Campaign Name
print(df.shape)
print(df["Campaign Name"].reset_index().drop_duplicates().shape)

"""Removing null values"""
print(df.isnull().sum())
print(df[df["# of Impressions"].isnull()])
null_date = df[df["# of Impressions"].isnull()].index[0]
df = df[df.index != null_date]  # Filtering out data will null values
print(df.shape)

"""Normalizing data"""
print(
    df.groupby("Campaign Name").agg({"Reach": "sum"})
)  # The data shows that 1 million more users are exposed to control group,
# which means that there might be biased due to non-equivalant distribution.
# Therefore, it is required to normalize the data to remove bias

norm_df = df.copy()
for col in [col for col in norm_df.columns if col not in ["Campaign Name", "Reach"]]:
    norm_df[col] = norm_df[col] / norm_df["Reach"]

print(norm_df)

# Analysis
# ---------
"""Seeing the difference between test and control groups"""
norm_control_df = norm_df[norm_df["Campaign Name"] == "Control Campaign"]
norm_test_df = norm_df[norm_df["Campaign Name"] == "Test Campaign"]


def bg_color(val):
    color = "red" if val < 0 else "green"
    return "color: %s" % color


control_desc_df = norm_control_df.describe()
test_desc_df = norm_test_df.describe()

my_styler = (
    (((test_desc_df - control_desc_df) / control_desc_df))
    .style.format(precision=2)
    .set_properties(**{"text-align": "right"})
    .map(bg_color)
)  # The data shows that test group performs better
my_styler

"""Joining test and control data sets to see the pattern over time"""
h_df = pd.merge(
    norm_control_df,
    norm_test_df,
    left_index=True,
    right_index=True,
    suffixes=("_cont", "_test"),
).drop(["Campaign Name_cont", "Campaign Name_test"], axis=1)

h_df = h_df.sort_index()
print(h_df)

for col in set([val.split("_")[0] for val in h_df.columns]):
    h_df[col + "_diff"] = (
        h_df[col + "_test"] - h_df[col + "_cont"]
    )  # / h_df[col + '_cont']

print(h_df)


"""Plotting """


def ab_test_plot(df, metric_def, metric_test, metric_diff):
    # Line plot
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[metric_def],
            name="Control",
            line_shape="linear",
            line=dict(color="royalblue", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[metric_test],
            name="Testing",
            line_shape="linear",
            line=dict(color="orange", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[metric_diff],
            name="Difference",
            line_shape="linear",
            line=dict(color="red", width=3),
            opacity=0.3,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Box plot
    fig.add_trace(
        go.Box(x=df[col + "_cont"], name="Control", boxpoints="all"), row=1, col=2
    )
    fig.add_trace(
        go.Box(x=df[col + "_test"], name="Testing", boxpoints="all"), row=1, col=2
    )

    fig.update_layout(
        title_text=metric_def.split("_")[0],
        yaxis=dict(title_text="Value"),
        yaxis2=dict(title_text="Difference"),
        width=1000,
        height=400,
    )

    fig.show()


for col in set([col.split("_")[0] for col in h_df.columns]):
    (
        ab_test_plot(h_df, col + "_cont", col + "_test", col + "_diff")
    )  # Remove anomolies if there are any

"""Removing anomoly data points"""
clean_h_df = h_df[~h_df.index.isin(["2019-08-12", "2019-08-19"])]
norm_df = norm_df[~norm_df.index.isin(["2019-08-12", "2019-08-19"])]
ab_test_plot(
    clean_h_df,
    "# of View Content_cont",
    "# of View Content_test",
    "# of View Content_diff",
)

"""pair plots"""
fig = sns.pairplot(norm_df.reset_index(), hue="Campaign Name", height=1.5, corner=False)
fig.map_upper(sns.kdeplot, levels=1)

"""Hyothesis Testing"""
# We will use two tests
# 1. Mann — Whitney U test
# 2. Kolmogorov — Smirnov
# Both are pair-wise comparison techniques.
# The null hypothesis in both cases is that both data point collections we are comparing
# (Testing and control) come from the same distribution, which would mean there is no change.
# Both are non-parametric tests, they don’t assume any underlying distribution.
# In both cases it assumes that samples are independent (This could be problematic as
# time series are by definition not independent, even so, it still being a widely used approach for our case)

results_list = list()

clean_df = clean_h_df.sort_index()

for col in set([col.split("_")[0] for col in clean_df.columns]):
    results_dict = dict()

    results_dict["metric"] = col
    _, results_dict["kstest"] = kstest(clean_df[col + "_cont"], clean_df[col + "_test"])
    _, results_dict["mwtest"] = mannwhitneyu(
        clean_df[col + "_cont"], clean_df[col + "_test"]
    )

    results_list.append(results_dict)

pprint(results_list)
results_list
