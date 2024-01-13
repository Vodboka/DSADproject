import pandas as pd
import numpy as np
import utilsDA as utils
import graphicsDA as graphics
import sklearn.metrics as metrics



try:
    file_1 = './dataIN/ProjectData.csv'

    table_1 = pd.read_csv(file_1, index_col=0)
    utils.replace_na_df(table_1)
    vars = np.array(table_1.columns)
    print(vars)
    var_categorical = vars[0:4]
    print(var_categorical)
    utils.codify(table_1, var_categorical)
    print(table_1)

    # Select the predictor variables and the discriminant variable
    var_p = vars[0:7]
    var_c = 'pollutant_id'
    print(var_p)
    print(var_c)

    x = table_1[var_p].values
    print(x)
    y = table_1[var_c].values
    print(y)
    n, m = np.shape(x)

    # Calculation of centers, frequencies, groups and dispersion
    g, ng, xg, sst, ssb, ssw = utils.dispersion(x, y)
    q = len(g)

    # Calculation of discriminating power of the predictor variables
    l_x, p_values = utils.discrim_power(ssb, ssw, n, q)
    discrimination_power_x = pd.DataFrame(
        data={
            'discrimination_power': l_x, 'p_value': np.around(p_values, 2)
        }, index=var_p
    )
    discrimination_power_x.to_csv('./dataOUT/Discrimination_power_x.csv')

    # Apply of linear discriminant analysis (LDA)
    alpha, l, u = utils.lda(sst, ssb, n, q)
    r = np.shape(u)[1]
    z = x @ u
    zg = xg @ u
    sst_t = utils.tabulation(X=sst, fileName='./dataOUT/SST.csv',
                             obsLabel=var_p,
                             varLabel=var_p)
    ssb_t = utils.tabulation(X=ssb, fileName='./dataOUT/SSB.csv',
                             obsLabel=var_p,
                             varLabel=var_p)
    z_index = ['z' + str(i) for i in range(1, len(l) + 1)]
    discrimination_power = pd.DataFrame(
        data={
            'Discrimination_power': l,
            'Discrimination_power(%)': l * 100 / sum(l),
            'Discrimination_power_cumulative(%)': np.cumsum(l) * 100 / sum(l),
            'alpha': alpha,
            'p_value': 1.0 - utils.sts.f.cdf(l, q - 1, n - q)
        }, index=z_index
    )
    discrimination_power.to_csv('./dataOUT/Discrimination_power.csv')
    utils.tabulation(X=u, varLabel=z_index, obsLabel=var_p,
                     fileName='./dataOUT/u.csv')
    utils.tabulation(X=z, varLabel=z_index, obsLabel=table_1.index,
                     fileName='./dataOUT/z.csv')

    # Compute the classification functions based on discriminant variable
    f, f0, f0_b = utils.classification_functions_z(z, zg, ng)
    utils.tabulation(X=f, fileName='./dataOUT/f.csv', varLabel=z_index, obsLabel=g)
    pd.DataFrame(data={'LDA': f0, 'Bayes': f0_b}, index=g).to_csv('./dataOUT/f0.csv')

    # Prediction based on LDA classification functions
    classification_LDA = utils.predict(z, f, f0, g)

    # Prediction based on Bayes classification functions 
    classification_Bayes = utils.predict(z, f, f0_b, g)
    cohen_kappa_lda = metrics.cohen_kappa_score(y, classification_LDA)
    cohen_kappa_bayes = metrics.cohen_kappa_score(y, classification_Bayes)
    classific_table = pd.DataFrame(data={
        var_c: y,
        'Prediction_LDA': classification_LDA,
        'Prediction_Bayes': classification_Bayes
    }, index=table_1.index)
    classific_table.to_csv('./dataOUT/Classification.csv')
    if r > 1:
        graphics.plot_scatter_groups(z[:, 0], z[:, 1], y, np.array(table_1.index),
                zg[:, 0], zg[:, 1], g, g,
                lx='z1 (' + str(l[0] * 100 / sum(l)) + ')',
                ly='z2 (' + str(l[1] * 100 / sum(l)) + ')')

    err_bayes = classific_table[classification_Bayes != y]
    err_lda = classific_table[classification_LDA != y]
    accuracy_global = pd.DataFrame(data=
        {'Accuracy': [100 - len(err_lda) * 100 / n, 100 - len(err_bayes) * 100 / n],
        'cohen_kappa': [cohen_kappa_lda, cohen_kappa_bayes]},
        index=['LDA', 'Bayes'])
    accuracy_global.to_csv('./dataOUT/Accuracy.csv')
    err_bayes.to_csv('./dataOUT/Err_Bayes.csv')
    err_lda.to_csv('./dataOUT/Err_LDA.csv')
    mat_conf_lda = utils.discrim_accuracy(y, classification_LDA, g)
    mat_conf_bayes = utils.discrim_accuracy(y, classification_Bayes, g)
    mat_conf_lda.to_csv('./dataOUT/Mat_conf_LDA.csv')
    mat_conf_bayes.to_csv('./dataOUT/Mat_conf_Bayes.csv')
    for i in range(r):
        plot_title = 'Distribution axis ' + str(i + 1) + '. Discrimination power(%):' \
                + str(round(discrimination_power.iloc[i, 1], 2))
        graphics.plot_distribution(z[:, i], y, g, title=plot_title)

    graphics.show()
except Exception as ex:
    print("Error!", ex.with_traceback(), sep="\n")
