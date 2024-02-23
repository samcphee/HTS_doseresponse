from dose_response import *

df = pd.read_csv("data.csv", index_col='sample_name')
df = process_doseresponse_df(df)
f = swarmplot_IC50results(df)
plt.show()
df = results_schema_output(df)
df.to_csv("processed_data.csv")
