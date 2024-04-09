#ref: https://www.jianshu.com/p/7379e015f1cf

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt
import pandas as pd
from pykrige.ok import OrdinaryKriging


df =  pd.read_excel('File_name.xlsx', sheet_name= 'Rn222')
# 从DataFrame中删除五行
# random_indices = np.random.choice(df.index, size=100, replace=False)
# df = df.drop(random_indices)



df_la = df['纬度'].values
df_lo = df['经度'].values
df_rn = np.log10(df['氡气'].values)
#df_rn = df['氡气'].values

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_points(x,y,z, theshold):
    filter_points_x = []
    filter_points_y = []
    filter_points_z = []
    for i in range(len(x)):
        max_index = i
        max_value = z[i]
        max_x = x[i]
        max_y = y[i]
        for j in range(len(x)):
            if i != j and distance ((x[i],y[i]), (x[j],y[j]))<theshold:
                if z[j] > max_value:
                    max_index = j
                    max_value = z[j]
                    max_x = x[j]
                    max_y = y[j]
        if i == max_index:
            filter_points_x.append(x[i])
            filter_points_y.append(y[i])
            filter_points_z.append(z[i])
    return filter_points_x, filter_points_y, filter_points_z

def filter_detlimit(z, limit):
    for i in range(len(z)):
        if z[i]<limit:
            z[i] = 0.001
    return z

# limit_f = -2
# rnf = filter_detlimit(df_rn, limit_f)

theshold = 0.0001

lof, lat, rnf = filter_points(df_lo,df_la,df_rn, theshold)

fig = plt.figure(figsize=(21,7))
plt.subplot(131)
plt.scatter(df_lo,df_la,c=df_rn, cmap= 'rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
colorbar1 = plt.colorbar()
colorbar1.set_label('log10(Rn222)')
# plt.colorbar()
# plt.subplot(132)
# plt.scatter(lof, lat, c=rnf)
# plt.colorbar()



Parameter = {'sill': 5, 'range': 10, 'nugget': 0.1}
Krin =  OrdinaryKriging(lof, lat, rnf, variogram_model= 'spherical',variogram_parameters=Parameter, nlags=6, coordinates_type='geographic')
olon = np.linspace(df_lo.min()-0.05, df_lo.max()+0.05, 100)
olat = np.linspace(df_la.min()-0.05, df_la.max()+0.05, 100)
data, ss = Krin.execute('grid', olon, olat)
olon, olat = np.meshgrid(olon, olat)

plt.subplot(132)
plt.contourf(olon, olat, data, 100, cmap= 'rainbow')
# b = plt.contour(olon, olat, data, 10, colors = 'black', linewidths = 1, linestyles = 'solid')
# plt.clabel(b, inline = True, fontsize = 5, fmt='%1.6f')
#plt.scatter(lof,lat,c=rnf, cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
colorbar2 = plt.colorbar()
colorbar2.set_label('log10(Rn222)')

plt.subplot(133)
plt.contourf(olon, olat, ss, 100, cmap= 'rainbow')
colorbar3 = plt.colorbar()
colorbar3.set_label('error')
plt.savefig('Rn222plot_kriging.png',dpi = 300)
#plt.show()
#print(data)
