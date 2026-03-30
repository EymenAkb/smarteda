import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SmartEDA:
    """
    SmartEDA provides automated exploratory data analysis (EDA)
    including data intuiton and visualization for pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    copy_data : bool, default=True
        Wheter using the dataframe or copying it.
    
    visualize_numeric : bool, default=False
        Whether to display numeric feature distributions.
    
    visualize_object : bool, default=False
        Whether to display categorical feature distributions.
    
    save_png : bool, default=False
        Save plots as PNG files.
    
    show_info : bool, default=False
        Give basic information about dataset.
    
    dataset_name : str, default='dataset'
        Create another directory for saving images
    """

    def __init__(self,
        df: pd.DataFrame = None,
        visualize_numeric: bool = False,
        visualize_object: bool = False,
        visualize_heatmap: bool = False,
        save_png_numeric: bool = False,
        save_png_object: bool = False,
        save_png_heatmap: bool = False,
        show_info: bool = False,
        dataset_name: str = "dataset"):
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')

        self.df = df
        self.visnum = visualize_numeric
        self.visobj = visualize_object
        self.visualize_heatmap = visualize_heatmap
        self.save_png_obj = save_png_object
        self.save_png_num = save_png_numeric
        self.save_png_hm = save_png_heatmap
        self.show_info = show_info

        self.dataset_name = dataset_name
        self.output_dir = os.path.join("eda_outputs", self.dataset_name)

        self.numeric_dir = os.path.join(self.output_dir, "numeric")
        self.object_dir = os.path.join(self.output_dir, "object")
        self.heatmap_dir = os.path.join(self.output_dir, "heatmap")

        if self.save_png_num:
            os.makedirs(self.numeric_dir, exist_ok=True)
        if self.save_png_obj:
            os.makedirs(self.object_dir, exist_ok=True)
        if self.save_png_hm:
            os.makedirs(self.heatmap_dir, exist_ok=True)

        try:
            if isinstance(df, pd.DataFrame):
                if df.empty:
                    raise ValueError("Data Frame must not be empty")
                else:
                    self.df = df
                    self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
                    self.object_cols = self.df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
                    self.render_plots()
            else:
                raise TypeError("DataFrame is not a Pandas DataFrame")
        except Exception as e:
            raise e


    @staticmethod
    def save_png_all(df, show_df_info=True,save_numeric=True, save_object=True, save_heatmap=True):
        SmartEDA(df=df, save_png_numeric=save_numeric, save_png_object=save_object, save_png_heatmap=save_heatmap, show_info=show_df_info)

    def render_plots(self):
        if self.show_info:
            self.info_show()
        
        if self.visnum or self.save_png_num:
            self.create_numeric()

        if self.visobj or self.save_png_obj:
            self.create_object()
            
        if self.vishm or self.save_png_hm:
            self.create_heatmap()
    

    @staticmethod
    def show_df_info(df):
        SmartEDA(df=df)._create_df_info_report()

    def info_show(self):
        self._create_df_info_report()
    
    def _create_df_info_report(self):
        print("\n=== DataFrame Info ===")
        self.df.info()
        #self.df.describe()
    
        print("\n=== Missing Values per Column ===")
        print(self.df.isnull().sum())

        print('\nColumn types:')
        print(f'Numeric columns: {", ".join(self.numeric_cols)}')
        print(f'Object columns: {", ".join(self.object_cols)}')


    @staticmethod
    def report_numeric(df, visualize=True, save_png=True, dataset_name="dataset"):
        SmartEDA(df=df, visualize_numeric=visualize, save_png_numeric=save_png, dataset_name=dataset_name)

    def create_numeric(self):
        self._create_numeric_vis()

    def _create_numeric_vis(self):
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        for column in self.numeric_cols:
            plt.figure(figsize=(10, 4))

            ax = sns.histplot(self.df[column], kde=True, bins=20, stat="count")

            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(int(height),
                            (p.get_x() + p.get_width()/2, height),
                            ha='center', va='bottom',
                            fontsize=8)

            plt.title(f"{column} - Distribution")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            plt.tight_layout()
            if self.save_png_num:
                plt.savefig(os.path.join(self.numeric_dir, f"{column}.png"))
            if self.visnum:
                plt.show()
            else:
                plt.close()


    @staticmethod
    def report_object(df, visualize=True, save_png=True, dataset_name="dataset"):
        SmartEDA(df=df, visualize_object=visualize, save_png_object=save_png, dataset_name=dataset_name)

    def create_object(self):
        self._create_object_vis()

    def _create_object_vis(self):
        self.object_cols = self.df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        for column in self.object_cols:
            count = self.df[column].astype(str).str.strip().value_counts().head(10)
            ax = count.plot(kind='bar', figsize=(12, 6))
            plt.title(f"{column} - Top 10 Values")
            plt.xlabel(column)
            plt.ylabel('Object count')
            plt.bar_label(ax.containers[0]) # Making bar label as count of the object
            if self.save_png_obj:
                plt.savefig(os.path.join(self.object_dir, f"{column}.png"))
            if self.visobj:
                plt.show()
            else:
                plt.close()
    

    @staticmethod
    def report_heatmap(df, visualize=True, save_png=True, dataset_name="dataset"):
        SmartEDA(df=df, visualize_heatmap=visualize, save_png_heatmap=save_png, dataset_name=dataset_name)

    def create_heatmap(self):
        self._create_heatmap_vis()

    def _create_heatmap_vis(self):
        if not self.numeric_cols:
            return "There is no numeric columns in the dataset."
        correlation = self.df[self.numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        if self.save_png_hm:
            plt.savefig(os.path.join(self.heatmap_dir, "heatmap.png"))
        if self.visualize_heatmap:
            plt.show()
        else:
            plt.close()
    