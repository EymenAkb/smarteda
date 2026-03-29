# SmartEDA

This repository was developed by **Eymen Akbulut** as part of automating EDA codes.

## Description

This program is an OOP (Object oriented programming) library code compact coding for EDA analysis.
This program is mostly on visualizing and summarizing data.

Some static methods are given below:

- `show_df_info` : Used for giving information, missing values and categories of columns as numeric or object columns.
- `visualize_numeric` : Used for saving png or directly visualizing numeric columns as histograms.
- `visualize_object` : Used for saving png or directly visualizing object columns as bar plots.
- `visualize_heatmap` : Showing correlation between numeric columns.

#### Note
*If you set `visualize=True` the library will iterate over each column.*
*If you don't want to see each of them one by one then you can set `visualize=False` and `save_png=True`*

## Required libraries

Some libraries which are essential for data analysis and visualizing:

- `Numpy` : Used for some fetching, esspecially with math
- `Pandas` : Interpreting, importing and everything about data.
- `Matplotlib` : For visualizing tasks.
- `Seaborn` : For visualizing tasks.
- `OS` : For better folder management.

## Usage

You can import the program if you put it in the same file in your code:

```python
Import SmartEDA as eda
```

docstring might be helpful throughout these processes.

## Note

Honestly i want to write something warm in between a serious description.
I hope this is useful library for your needings. 
*This library doesn't summarize the code as pdf only for intuition.*
*This is only a prototype so it can only work for numeric and categorical features.*