import pandas as pd
import functions as fun
import pca.PCA as pca
import graphics as g

# Assuming your data is in a CSV file with the structure you provided earlier
file_path = './dataIN/ProjectData.csv'
table = pd.read_csv(file_path, index_col=0)

# create a list of useful variables
# vars = table.columns[1:]
vars = table.columns.values[5:]
# vars = list(table.columns.values[1:])
print(vars, type(vars))

# create a list of observations
obs = table.index.values
print(obs, type(obs))

# no. of variables
m = vars.shape[0]
print(m)
# no. of observations
n = len(obs)
print(n)

# create the matrix X of observed variables
X = table[vars].values
print(X.shape, type(X))

# standardise the X matrix
Xstd = fun.standardize(X)
print(Xstd.shape)
# save the standardised matrix into CSV file
Xstd_df = pd.DataFrame(data=Xstd, index=obs,
                       columns=vars)
print(Xstd_df)
Xstd_df.to_csv('./dataOUT/Xstd.csv')

# instantiate a PCA object
modelPCA = pca.PCA(Xstd)
alpha = modelPCA.getEigenValues()

g.principalComponents(eigenvalues=alpha)
# g.show()

# extract the principal components
prinComp = modelPCA.getPrinComp()
# save principal components into a CSV file
components = ['C'+str(j+1) for j in range(prinComp.shape[1])]
prinComp_df = pd.DataFrame(data=prinComp, index=obs,
                           columns=components)
prinComp_df.to_csv('./dataOUT/PrinComp.csv')

# extract the factor loadings
factorLoadings = modelPCA.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=vars,
                                 columns=components)
# save the factor loadings into a CSV file
factorLoadings_df.to_csv('./dataOUT/factorLoadings.csv')
g.correlogram(matrix=factorLoadings_df, title='Correlogram of factor loadings')
# g.show()

# extract teh scores
scores = modelPCA.getScores()
scores_df = pd.DataFrame(data=scores, index=obs, columns=components)
# save the scores
scores_df.to_csv('./dataOUT/Scores.csv')
g.intensity_map(matrix=scores_df, title='Standardized principal components (scores)')
# g.show()

# extract the quality of points representation
qualObs = modelPCA.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, index=obs, columns=components)
# save the quality of points representation
qualObs_df.to_csv('./dataOUT/QualityObservations.csv')
g.intensity_map(matrix=qualObs_df, title='Quality of points representation')
# g.show()

# extract the observations contribution to the axes' variance
contribObs = modelPCA.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs, index=obs, columns=components)
# save the observation contribution to the axes variance
contribObs_df.to_csv('./dataOUT/ObservationContributionToTheAxesVariances.csv')
g.intensity_map(matrix=contribObs_df, title="Observations contribution to the axes' variance")
# g.show()

# extract the communalities
commun = modelPCA.getCommun()
commun_df = pd.DataFrame(data=commun, index=vars, columns=components)
# save the communalities
commun_df.to_csv('./dataOUT/Communalities.csv')
g.intensity_map(matrix=commun_df, title='Communalities')
g.show()
