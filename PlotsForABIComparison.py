'''
Creates Plots for comparing different ABI data. Requires ABI data to be saved out to pickle files, which is done with the BinnedCountPredictionModel.py script
Bins can be changed to see if there are real differences between ABI for different bins of lightning counts
'''
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
with open('/glade/u/home/gwallach/goes16ci/train_data_scaled.pkl','rb') as f:
    x = pickle.load(f)
with open('/glade/u/home/gwallach/goes16ci/train_counts.pkl','rb') as f:
    y = pickle.load(f)
x.shape
#find data for 1-10 lightning counts
lowcounts = np.where(np.logical_and(y[:]>=1, y[:]<=10))
lowcountsy = y[lowcounts]
lowcountsy.shape
lowcountsx = x[lowcounts]
lowcountsx.shape
#find counts from 10-50
midcounts = np.where(np.logical_and(y[:]>10, y[:]<=50))
midcountsx = x[midcounts]
print(midcountsx.shape)
#find counts from 50-100
midhighcounts = np.where(np.logical_and(y[:]>50, y[:]<=100))
midhighcountsx = x[midhighcounts]
print(midhighcountsx.shape)
#find counts for 100-up
highcounts = np.where(y[:]>100)
highcountsx = x[highcounts]
print(highcountsx.shape)
lowcountsx = lowcountsx.reshape(41905 * 32 * 32, 4)
lowcountsx.shape
midcountsx = midcountsx.reshape(8024 * 32 * 32, 4)
midhighcountsx = midhighcountsx.reshape(1268 * 32 * 32, 4)
highcountsx = highcountsx.reshape(772 * 32 * 32, 4)
plt.figure(figsize=(12,12))
for i,band in enumerate(['Band: 08', 'Band: 09', 'Band: 10', 'Band: 14']):
    sns.kdeplot(lowcountsx[:,i], label=band, shade=True).set_title(
            'Mean Patch Distributions With 1-10 Lightning Counts', fontsize=22)
plt.savefig('Patch Distribution With 1-10 Lightning Counts')

plt.figure(figsize=(12,12))
for j,band in enumerate(['Band: 08', 'Band: 09', 'Band: 10', 'Band: 14']):
    sns.kdeplot(midcountsx[:,j], label=band, shade=True).set_title(
            'Patch Distributions With 10-50 Lightning Counts', fontsize=22)
plt.savefig('Patch Distribution With 10-50 Lightning Counts')

plt.figure(figsize=(12,12))
for k,band in enumerate(['Band: 08', 'Band: 09', 'Band: 10', 'Band: 14']):
    sns.kdeplot(midhighcountsx[:,k], label=band, shade=True).set_title(
            'Patch Distributions With 50-100 Lightning Counts', fontsize=22)
plt.savefig('Patch Distribution With 50-100 Lightning Counts')

plt.figure(figsize=(12,12))
for l,band in enumerate(['Band: 08', 'Band: 09', 'Band: 10', 'Band: 14']):
    sns.kdeplot(highcountsx[:,l],label=band, shade=True).set_title(
            'Mean Patch Distributions With 100 and Up Lightning Counts', fontsize=22)
plt.savefig('Patch Distribution With 100 and Up Lightning Counts')
    

