# Team Challege: Toolbox I

### Project Overview

This repository contains the deliverables for the first part of the team challenge, which focuses on building a toolbox module for simplifying the creation of Machine Learning (ML) models. The goal of this challenge is to implement a set of functions that aid in feature analysis and selection for a Machine Learning problem, and compile them into a Python script called `toolbox_ML.py`.

This module will be used in the second part of the challenge to solve a practical Machine Learning problem.

### Deliverables

1. **Python script**: `toolbox_ML.py`

    - This script contains the implementations of the required functions.
    - Each function is properly commented and includes a **docstring** with a description of its  usage, parameters, and return values.
    
2. **Example code**:

    - The group has designed an example that demonstrates the usage of the functions.
    - This example is provided in a separate notebook or script in the repository, showcasing how the functions can be applied to a dataset.
    
3. **Presentation**:

      - A brief presentation that describes the code and demonstrates the example.
      - This presentation explains the motivation behind the design decisions and provides insights into the utility of the functions.

### Functions implemented

The following functions have been developed as part of the `toolbox_ML.py` module:

1. `describe_df(df)`
   - **Purpose**: Analyzes a dataframe and returns a summary of each column's data type, percentage of missing values, number of unique values, and percentage of cardinality.
    - **Returns**: A dataframe summarizing the characteristics of each column.
      
2. `tipifica_variables(df, umbral_categoria, umbral_continua)`
    - **Purpose**: Suggests a variable type (binary, categorical, continuous, or discrete) based on the cardinality of each column in the dataframe.
    - **Returns**: A dataframe with two columns: nombre_variable (variable name) and tipo_sugerido (suggested type).
      
3. `get_features_num_regression(df, target_col, umbral_corr, pvalue = None)`
    - **Purpose**: Identifies numerical columns whose correlation with the target column exceeds a specified threshold. Optionally, it can also filter by statistical significance.
    - **Returns**: A list of numerical columns meeting the correlation and statistical significance criteria.
      
4. `plot_features_num_regression(df, target_col = "", columns = [], umbral_corr = 0, pvalue = None)`
    - **Purpose**: Generates pair plots of numerical features that meet the correlation and statistical significance criteria, using `target_col` as a reference. It splits large sets of columns into multiple pair plots if needed.
    - **Returns**: The list of columns that meet the plotting criteria.
      
5. `get_features_cat_regression(df, target_col, pvalue = 0.05)`
    - **Purpose**: Identifies categorical columns that have a statistically significant relationship with the target column, using appropriate hypothesis tests.
    - **Returns**: A list of categorical columns that meet the statistical significance criteria.
      
6. `plot_features_cat_regression(df, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False)`
    - **Purpose**: Generates histograms of categorical features grouped by the values of the target column, for features that meet the statistical significance criteria.
    - **Returns**: The list of columns that meet the plotting criteria.

### Next steps

This module serves as the foundation for the second part of the challenge, where the team will apply these functions to a real Machine Learning problem, further refining and expanding the toolbox as needed.
