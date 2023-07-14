
def eda(df, term):
    if term=='others':
        variables = list(df.filter(regex='^(?!.*rev|.*qty|.*mou)').columns)
        analysis_num = 2
    else:
        variables = list(df.filter(like=term).columns)
        analysis_num = 1

    if analysis_num == 1:
        # Create a scatter plot for each variable compared to churn
        # Calculate the number of subplots needed
        num_plots = len(variables)

        # Create a subplot matrix
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            sns.violinplot(x='churn', y=var, data=df, ax=ax)
            ax.set_xlabel('churn')
            ax.set_ylabel(var)
        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_churn_relation.png")

        # Analyze the distribution of the selected variables
        # Create a histogram for each variable
        # Create a subplot matrix
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            df[var].hist(bins=20, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_distribution.png")

    elif analysis_num ==2:
        df_rest_cols = df.filter(regex='^(?!.*rev|.*qty|.*mou)')

        df_rest_cols.index = df_rest_cols['Customer_ID']
        df_rest_cols = df_rest_cols.drop('Customer_ID', axis=1)

        # select only numerical variables:
        num_cols = df_rest_cols.select_dtypes(include='number')
        variables = num_cols.columns

        # Crear un gr�fico de dispersi�n para cada variable comparada con churn
        # Calcular el n�mero de subtramas necesarias
        num_plots = len(variables)

        # Crear una matriz de subtramas
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            sns.violinplot(x='churn', y=var, data=df, ax=ax)
            ax.set_xlabel('churn')
            ax.set_ylabel(var)
        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_numeric_churn_relation.png")

        # Analizar la distribuci�n de las variables 'rev'
        # Crear un histograma para cada variable
        # Crear una matriz de subtramas
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            df[var].hist(bins=20, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_numeric_distribution.png")

        # select only categorical variables:
        cat_cols = df_rest_cols.select_dtypes(include='object')
        variables = cat_cols.columns

        # Crear un gr�fico de dispersi�n para cada variable comparada con churn
        # Calcular el n�mero de subtramas necesarias
        num_plots = len(variables)

        # Crear una matriz de subtramas
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            stacked_data = df.groupby(['churn', var])[var].count().unstack('churn').fillna(0)
            stacked_data = stacked_data.divide(stacked_data.sum(axis=1), axis=0)
            stacked_data.plot(kind='bar', stacked=False, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Proportion')
            ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_categorical_churn_relation.png")

        # Analizar la distribuci�n de las variables 'rev'
        # Crear un histograma para cada variable
        # Crear una matriz de subtramas
        num_rows = (num_plots // 4) + 1
        fig, axs = plt.subplots(num_rows, 4, figsize=(16, num_rows*4))

        for i, var in enumerate(variables):
            ax = axs[i//4, i%4]
            df[var].hist(bins=20, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"SDG-Case-Study//plots//eda_{term}_categorical_distribution.png")

def drop_outliers(df, term):
    variables = list(df.filter(like=term).columns)

    for col in variables:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        iqr = q3 - q1
        df = df.loc[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    
    return df