import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SmartEDA:
    """
    SmartEDA provides automated exploratory data analysis (EDA)
    including data cleaning and visualization for pandas DataFrames.

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
        df: pd.DataFrame,
        copy_data: bool = True,
        visualize_numeric: bool = False,
        visualize_object: bool = False,
        visualize_heatmap: bool = False,
        save_png_numeric: bool = False,
        save_png_object: bool = False,
        save_png_heatmap: bool = False,
        show_info: bool = False,
        dataset_name: str = "dataset"):
        
        """
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        copy_data : bool, default=True
            Whether to work on a copy of the dataframe.
        visualize_numeric : bool, default=False
            Plot numeric feature distributions.
        visualize_object : bool, default=False
            Plot categorical feature distributions.
        save_png : bool, default=False
            Save generated plots as PNG files.
        show_info : bool, default=False
            Display dataframe info and missing value summary.
        dataset_name : str, default='dataset'
            Create another directory for saving images
        """
        
        # - - - Using modern plot style - - - #
        plt.style.use('seaborn-v0_8') 
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must not be empty")

        self.visnum = visualize_numeric
        self.visobj = visualize_object
        self.vishm = visualize_heatmap
        self.save_png_obj = save_png_object
        self.save_png_num = save_png_numeric
        self.save_png_hm = save_png_heatmap
        self.show_info = show_info
    
        self.df = df.copy(deep=True) if copy_data else df
        
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.object_cols = self.df.select_dtypes(include=["object", "category", "str"]).columns.tolist()

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
        self.render_plots()

    def render_plots(self):
            if self.visnum or self.save_png_num:
                self.vis_numeric()

            if self.visobj or self.save_png_obj:
                self.vis_obj()
            
            if self.vishm or self.save_png_hm:
                self.create_heatmap()

    def show_df_info(self):
        # Giving information to user using terminal as interface

        print("\n=== DataFrame Info ===")
        self.df.info()
    
        print("\n=== Missing Values per Column ===")
        print(self.df.isnull().sum())

        print('\nColumn types:')
        print(f'Numeric columns: {", ".join(self.numeric_cols)}')
        print(f'Object columns: {", ".join(self.object_cols)}')
    
    def vis_numeric(self):
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
        
    def vis_obj(self):
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
    
    def create_heatmap(self):
        if not self.numeric_cols:
            return "There is no numeric columns in the dataset."
        correlation = self.df[self.numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        if self.save_png_hm:
            plt.savefig(os.path.join(self.heatmap_dir, "heatmap.png"))
        plt.show()
