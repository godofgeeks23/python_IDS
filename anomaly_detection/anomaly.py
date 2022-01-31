import pandas as pd
import plotly.express as px
import kaleido as kld

data = pd.read_csv('org_dataset\Subdataset1.csv')
data['DATETIME'] = pd.to_datetime(data['DATETIME'], format='%d/%m/%y %H')
print(data.head())

# # create list of columns in the csv file
# cols = data.columns.tolist()
# cols.remove('DATETIME')
# for col in cols:
#     colname48 = "MA48_" + col
#     data[colname48] = data[col].rolling(48).mean()
#     colname336 = "MA336_" + col
#     data[colname336] = data[col].rolling(336).mean()
#     fig = px.line(data, x="DATETIME", y=[col, colname48, colname336], title=col, template = 'plotly_dark')
#     imgname = "images/" + col + ".png"
#     fig.write_image(imgname)



data['MA48'] = data['L_T1'].rolling(48).mean()
data['MA336'] = data['L_T1'].rolling(336).mean()# plot 
fig = px.line(data, x="DATETIME", y=['L_T1', 'MA48', 'MA336'], title='L_T1', template = 'plotly_dark')
fig.show()
# fig.write_image("images/L_T1.png")
