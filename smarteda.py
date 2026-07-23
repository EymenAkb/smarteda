import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Literal, get_args
import warnings

try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

hande_iqr_methods = Literal["ignore", "remove"]

class Data:    
    @staticmethod
    def load_data(df: pd.DataFrame | str | None = None, index_col: str | None = None, date_col: str | None = None, can_return_none: bool = False) -> pd.DataFrame:
        if isinstance(df, str):
            try:
                data = pd.read_csv(df)

            except Exception as e:
                if can_return_none:
                    return None
                else:
                    raise RuntimeError(f"Error: '{e}' occurred during the run.")
        
        elif isinstance(df, pd.DataFrame):
            data = df

        else:
            if can_return_none:
                return None
            else:
                raise ValueError(f"Couldn't read the dataframe")
        
        if index_col:
            try:
                data.set_index(index_col, inplace=True)
            except Exception as e:
                if can_return_none:
                    return None
                else:
                    raise RuntimeError(f"Error: '{e}' occurred during the run.")

        if date_col:
            try:
                data[date_col] = pd.to_datetime(data[date_col])
            except Exception as e:
                if can_return_none:
                    return None
                else:
                    raise RuntimeError(f"Error: '{e}' occurred during the run.")
        
        return data
    

    @staticmethod
    def calculate_iqr(series: pd.Series):
        q_25 = series.quantile(0.25)
        q_75 = series.quantile(0.75)
        iqr = q_75 - q_25
        
        low = q_25 - (1.5 * iqr)
        high = q_75 + (1.5 * iqr)
        return (low, high)


    @staticmethod
    def return_iqr(df: pd.DataFrame = None, column: str | list | tuple = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataFrame isn't a pandas DataFrame object.")
        
        if isinstance(column, str):
            column = [column]

        if not isinstance(column, (list, tuple)):
            raise TypeError(f"The column isn't in the specified types")
        
        data = df.copy()

        for col in column:
            if col in data.columns:
                low, high = Data.calculate_iqr(data[col])
                data = data[(data[col] >= low) & (data[col] <= high)]
            else:
                warnings.warn(f"Couldn't find the column: {col} in the DataFrame.")
        return data


class SmartEDA:
    """
    SmartEDA provides automated exploratory data analysis (EDA)
    including data intuiton and visualization for pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    index_col : str [Optional], default=None
        Index column of the DataSet.

    date_column : str [Optional], default=None
        Date column in the Dataset.

    dataset_name : str, default='Dataset'
        Base directory name for pngs.

    visualize_numerical : bool, default=False
        Whether to display numeric feature distributions.

    save_numerical_figures : bool, default=False
        Whether to save numerical plots as PNG files.
    
    visualize_categorical : bool, default=False
        Whether to display categorical feature distributions.

    save_categorical_figures : bool, default=False
        Whether to save categorical plots as PNG files.
    
    visualize_heatmap : bool, default=False
        Whether to display correlation between the numerical values.
    
    save_heatmap_figures : bool, default=False
        Whether to save heatmap plot as PNG file.

    handle_iqr : ["ignore", "remove"], default="ignore"
        How to handle outlier values.
    
    show_info : bool, default=False
        Give basic information about the dataset.
    """
    def __init__(self, df: pd.DataFrame | str | None = None, index_col=None, date_column=None, dataset_name: str ="Dataset", 
                 visualize_numerical: bool = False, save_numerical_figures: bool = False, visualize_categorical: bool = False, save_categorical_figures: bool = False,
                 visualize_heatmap: bool = False, save_heatmap_figure: bool = False, handle_iqr: hande_iqr_methods = "ignore", show_info: bool =False):
        if not handle_iqr.lower().strip() in get_args(hande_iqr_methods):
            warnings.warn(f"IQR method isn't in the spesified types: {get_args(hande_iqr_methods)}, continuing with ignore.")
            self.handle_iqr = "ignore"
        else:
            self.handle_iqr = handle_iqr
        self.df = Data.load_data(df, index_col, date_column, can_return_none=True)
        self.date_column = date_column
        self.visualize_numeric = visualize_numerical
        self.save_num_figs = save_numerical_figures
        self.visualize_category = visualize_categorical
        self.save_cat_figs = save_categorical_figures
        self.visualize_hm = visualize_heatmap
        self.save_hm_fig = save_heatmap_figure
        self.dataset_name = dataset_name
        self.show_info = show_info

        if self.df:
            self._run()

    def _run(self):
        if self.df.empty:
            raise ValueError(f"DataFrame must not be empty.")
        
        self.numerical_columns = self.df.select_dtypes(include=np.number).columns.to_list()
        self.categorical_columns = self.df.select_dtypes(exclude=np.number).columns.to_list()

        if self.handle_iqr == "remove":
            self.df = Data.return_iqr(self.df, self.numerical_columns)
        self._render_plots()

    def _render_plots(self):
        if self.show_info:
            self.show_df_info()

        if self.visualize_numeric or self.visualize_category or self.visualize_hm:
            self._setup_paths()

        if self.visualize_numeric or self.save_num_figs:
            self._create_numerical()
        
        if self.visualize_category or self.save_cat_figs:
            self._create_categorical()

        if self.visualize_hm or self.save_hm_fig:
            self._create_heatmap()
        
    @staticmethod
    def render_plots(**kwargs):
        if not handle_iqr in get_args(hande_iqr_methods):
            warnings.warn(f"IQR method isn't in the spesified types: {get_args(hande_iqr_methods)}, continuing with ignore.")
            handle_iqr = "ignore"
        SmartEDA(**kwargs)

    @staticmethod
    def save_all_figures(**kwargs):
        SmartEDA(
            save_numerical_figures=True, 
            save_categorical_figures=True, 
            save_heatmap_figure=True, 
            **kwargs
        )

    @staticmethod
    def visualize_all(**kwargs):
        SmartEDA(
            visualize_numerical=True, 
            visualize_categorical=True, 
            visualize_heatmap=True, 
            **kwargs
        )

    def _show_df_info(self):
        print(self)

    @staticmethod
    def show_df_info(**kwargs):
        SmartEDA(**kwargs, show_info=True)
        
    def _create_numerical(self):
        if not self.numerical_columns:
            warnings.warn(f"There are no numerical columns in the dataset.", UserWarning)
            return 
        for column in self.numerical_columns:
            plt.figure(figsize=(10, 4))

            ax = sns.histplot(self.df[column], kde=True, bins=20, stat="count")

            plt.title(f"{column} - Distribution")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            plt.tight_layout()

            if self.save_num_figs:
                plt.savefig(os.path.join(self.numerical_path, f"{column}.png"))

            if self.visualize_numerical:
                plt.show()

            plt.close()

    @staticmethod
    def visualize_numerical(**kwargs):
        SmartEDA(**kwargs, visualize_numerical=True)


    def _create_categorical(self):
        if not self.categorical_columns:
            warnings.warn(f"There are no categorical columns in the dataset.", UserWarning)
            return 
        for column in self.categorical_columns:
            count = self.df[column].astype(str).str.strip().value_counts().head(10)

            ax = count.plot(kind='bar', figsize=(12, 6))

            plt.title(f"{column} - Distribution")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.bar_label(ax.containers[0])

            if self.save_cat_figs:
                plt.savefig(os.path.join(self.categorical_path, f"{column}.png"))

            if self.visualize_category:
                plt.show()

            plt.close()
    
    @staticmethod
    def visualize_categorical(**kwargs):
        SmartEDA(**kwargs, visualize_categorical=True)
        
    def _create_heatmap(self):
        if not self.numerical_columns:
            warnings.warn(f"There are no numerical columns in the dataset.", UserWarning)
            return
        ax = sns.heatmap(self.df[self.numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Heatmap correlation")
        if self.save_hm_fig:
            plt.savefig(os.path.join(self.heatmap_path, "heatmap.png"))
        if self.visualize_hm:
            plt.show()
        
        plt.close()
  
    @staticmethod
    def visualize_heatmap(**kwargs):
        SmartEDA(**kwargs, visualize_heatmap=True)

    def _setup_paths(self):
        self.base_path = os.path.join(".", "eda_output", self.dataset_name)
        if self.save_num_figs:
            self.numerical_path = os.path.join(self.base_path, "numerical")
            os.makedirs(self.numerical_path, exist_ok=True)
        if self.save_cat_figs:
            self.categorical_path = os.path.join(self.base_path, "categorical")
            os.makedirs(self.categorical_path, exist_ok=True)
        if self.save_hm_fig:
            self.heatmap_path = os.path.join(self.base_path, "heatmap")
            os.makedirs(self.heatmap_path, exist_ok=True)

    def __call__(self, df: pd.DataFrame | str = None , index_col: str = None, date_column: str = None, dataset_name: str = "Dataset"):
        if df is not None:
            if index_col is not None:
                self.index_col = index_col
            if date_column is not None:
                self.date_column = date_column
            self.dataset_name = dataset_name
            self.df = Data.load_data(df, self.index_col, self.date_column, can_return_none=False)
            
        if self.df is None or self.df.empty:
            raise RuntimeError("No valid DataFrame provided to run SmartEDA.")
            
        self._run()
        return self

    def __len__(self):
        return len(self.numerical_columns + self.categorical_columns)
    
    def __str__(self):
        result = f"""
Dataset details:

Dataset name: {self.dataset_name}
Total column numbers: {len(self.numerical_columns + self.categorical_columns)}

----------------

Empty rows per column:
{self.df.isnull().sum()}

----------------

Dataset information:
{self.df.info()}

----------------

Some samples from the dataset:
{self.df.head()}

"""