# SmartEDA

This repository was developed by **Eymen Akbulut** as part of automating EDA codes.

## Description

This program is an OOP (Object oriented programming) library code compact coding for EDA analysis.
This program is mostly on visualizing and summarizing data.

Some static methods are given below:

- `show_df_info` : Used for giving information, missing values and categories of columns as numeric or object columns.
- `report_numeric` : Visualizing numeric columns and saving histograms as png. (you can adjust saving or visualizing)
- `report_object` : Visualizing object columns and saving bar plots as png. (you can adjust saving or visualizing)
- `report_heatmap` : Visualizing and saving heatmap as png. (you can adjust saving or visualizing)
- `save_png_all` : Saving png for numerical, categorical columns and also saving heatmap as correlation of numeric columns.

#### Note
*If you set `visualize=True` the library will iterate over each column.*
*If you don't want to see each of them one by one then you can set `visualize=False` and `save_png=True`*
*In standart way static reports will automatically save pngs and visualize the plots. Except `save_png_all`*
*`save_png_all` method is adjustable for any column type you want.*

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

*This library doesn't summarize the code as pdf only for intuition.*
*This is only a prototype so it can only work for numeric and categorical features.*