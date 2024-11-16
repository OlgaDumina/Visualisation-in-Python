import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 
          'SEP', 'OCT', 'NOV', 'DEC']
stations = ['ID', 'Lat', 'Lon', 'Elev', 'Station_name', 'BI']
months_name = {
    'JAN' : 'January', 'FEB' : 'February', 'MAR' : 'March', 'APR' : 'April', 
    'MAY' :'May', 'JUN' :'June', 'JUL' : 'July', 'AUG' : 'August', 
    'SEP' :'September', 'OCT' : 'October', 'NOV' : 'November', 
    'DEC' : 'December'
}

# define colors
GRAY1, GRAY2, GRAY3 = '#231F20', '#414040', '#555655'
GRAY4, GRAY5, GRAY6 = '#646369', '#76787B', '#828282'
GRAY7, GRAY8, GRAY9 = '#929497', '#A6A6A5', '#BFBEBE'
BLUE1, BLUE2, BLUE3, BLUE4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
RED1, RED2 = '#C3514E', '#E6BAB7'
GREEN1, GREEN2 = '#0C8040', '#9ABB59'
ORANGE1 = '#F79747'

PSEUDOMAP_N=6       # number of intervals for pseudomap plotting
# colors for each boundary value in pseudomap
PSEUDOMAP_COLORS = ['purple','navy','blue','lime',
                    'yellow','orange','red','red']  

def select_data (input_file_name, code_string):
    '''
    Select data from input file for the stations from one country
    input_file_name : the name of input file
    code_string : the first letters of the station name in the country
    return : DataFrame with selected data
    '''
    # Variables for data storage
    coords = []
    data_blocks = []
    temp_block = []

    # reading file with data
    with open(input_file_name, 'r') as file:
        line = file.readline()

        while line:         # Ukrainian station
            if  line.startswith(code_string):
                # Add coordinates to the list
                coords.append(line.split())
                line = file.readline()
                if not line[0].isdigit(): 
                    data_blocks.append(temp_block)
                    temp_block = []
                    continue
                # if it is temperature data
                while line[0].isdigit():
                    # Add a row with data by year and temperature 
                    # to the current block
                    temp_block.append(line.split())
                    line = file.readline()
                data_blocks.append(temp_block)
                temp_block = []
                continue
            line = file.readline()
            

    # Add the last block of temperature data, if it exist
    if temp_block:
        data_blocks.append(temp_block)
    # Create the final DataFrame
    final_df = pd.DataFrame()

    # Convert each block
    for i, block in enumerate(data_blocks):
        df_block = pd.DataFrame(block, columns = ['Year'] + months)
        coord_info = pd.DataFrame([coords[i]] * len(df_block), 
                                  columns = stations)
        merged_df = pd.concat([coord_info, df_block], axis = 1)
        final_df = pd.concat([final_df, merged_df])

    
    return final_df

def preprocess (df):
    '''
    Convert data to correct numerical format 
    df : dataframe for processing 
    '''

    # Convert Latitude, Longitude, Elevation to float
    df[['Lat','Lon']] = df[['Lat','Lon']].apply(
        lambda x: x.str.replace(',','.').astype(float))
    df['Elev'] = df['Elev'].astype(float)
    df['Year'] = df['Year'].astype(int)

    # Process only month columns
    for col in months:
        # Convert temperature from string to float and divide by 100
        df[col] = df[col].astype(float) / 100

    return df

def history (df, year_start, year_end, step):
    '''
    Count and plot monthly average number of measurements 
    in time intervals
    returns a DataFrame with the calculated data
    df : DataFrame with temperature data
    year_start, year_end : limits of the entire time range
    step : length of the intervals 
    '''
    # Upper boundary of the interval [1900, 1920, ... 2000, 2020]
    YEAR_MAX = np.arange(year_start + step, year_end, step) 

    # DataFrame with the calculated data
    count_df = pd.DataFrame({'YEAR_MAX': YEAR_MAX})

    # Calculate temperature values by period
    for i, year in enumerate(count_df['YEAR_MAX']):
        filtered_df = df[(df['Year'] >= year - step) & (df['Year'] < year)]
        count_df.loc[i, 'count_err'] = \
            (filtered_df[months] == -99.99).sum().sum() / 240
        count_df.loc[i, 'count_station'] = \
            (filtered_df[months] > -99.99).count().sum() / 240
    # set 'YEAR_MAX' as an index
    count_df.set_index('YEAR_MAX', inplace = True)  

    ax = count_df.plot(kind = 'bar', figsize = (10, 6))
    ax.set_title(
      f'Monthly average number of measurements for the previous {step} years')
    ax.set_xlabel('Year')
    plt.show()

    count_df.reset_index(inplace = True)  # restore 'YEAR_MAX' as a column

    return count_df

def history_grouped(df):
    '''
    plot grouped bar chart of the average monthly 
    number of measurements and errors
    df : DataFrame with the calculated data
    '''

    # calculate the relative number of failed measurements  
    df['procent_err'] = df['count_err'] / df['count_station'] * 100

    fig, ax = plt.subplots(figsize = (12, 6))
    # tune the subplot layout by setting sides of the figure
    fig.subplots_adjust(left = 0.103, right = 0.7, top = 0.881, bottom = 0.096)

    fig.suptitle('Total and Failed Measurements in the Ukraine Over Years', 
                 color = GRAY2, fontsize = 22 )

    bar_width = 13

    # Total measurements
    ax.bar(df['YEAR_MAX'], df['count_station'], 
           width = bar_width, color = BLUE2, label = 'Total Measurements')

    # Failed measurements placed against the background of the total
    ax.bar(df['YEAR_MAX'], df['procent_err'], 
           width = bar_width, color = ORANGE1, label = 'Failed Measurements')

    # Add value labels above columns
    for i in range(len(df)):
        
        ax.text(df['YEAR_MAX'][i], df['procent_err'][i] + 1, 
                str(df['procent_err'][i].round(2)) + '%', 
                ha = 'center', color = GRAY2)

    # add text labeling 
    ax.text(2040, 50, 'Average number of \naktive stations \nper month', 
            fontsize = 16, color = BLUE2)
    ax.text(2040, 7, 'Relative number of \nfailed measurements \nper month', 
            fontsize = 16, color = ORANGE1)

    step = df['YEAR_MAX'][1] - df['YEAR_MAX'][0]

    # Shifted column to create range start values
    df['Year_Start'] = df['YEAR_MAX'].shift(1, fill_value = 
                                            df['YEAR_MAX'].iloc[0] - step)

    # Array of ranges as strings
    year_ranges = df.apply(lambda row: \
        f"{int(row['Year_Start'])}-{int(row['YEAR_MAX'])}", axis = 1).tolist()

    # set properties for axes object (ticks for all issues with labels)
    ax.tick_params(axis = 'both', colors = GRAY6)
    ax.tick_params(axis = 'x', length = 0)
    ax.set_xticks(df['YEAR_MAX'])
    ax.set_xticklabels(year_ranges)
    
    # remove chart border
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(GRAY4)
    ax.spines[['bottom', 'left']].set_linewidth(1.5)

    # Add a line to an axis
    ax.plot([2010, 2010], [0, 75], color = GRAY6)
    ax.text(2001, 75, '2014', fontsize = 24, color = GRAY6)
    # Add informative text
    ax.text(2025, 73, 'Der Beginn der russischen \nInvasion in der Ukraine',
        fontsize = 16, color = GRAY4)

    plt.show()

def replace_outliers(group, outlier = -99.99):
    '''
    Replace outliers with an average value
    group : data to process
    outlier : value of the outliers
    '''

    for col in months:
        # Calculate the average for each group
        mean_value = group[group[col] != outlier][col].mean().round(2) 
        # Replace -99.99 with the average value 
        group[col] = group[col].replace(outlier, mean_value)   
    return group

def IQR (df, ax):
    '''
    plot Interquartile Range by Tufte
    df : Dataframe
    ax : axes
    '''

    # calculate percentiles
    Q1 = np.percentile(df, 25)
    Q2 = np.percentile(df, 50)
    Q3 = np.percentile(df, 75)
    IQR = Q3 - Q1

    # outlier boundary
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bounded_df = df[(df >= lower_bound) & (df <= upper_bound)]
    outliers = df[(df < lower_bound) | (df > upper_bound)]

    e = (Q3 - Q1) / 50      # line break at the mean value

    y_positions = [0, -0.5, -0.5, 0]
    
    # Start and end points of segments along the X-axis
    x_start = [bounded_df.min(), Q1, Q2 + e, Q3]
    x_end = [Q1, Q2 - e, Q3, bounded_df.max()]
    
    # Draw several horizontal segments
    ax.hlines(y = y_positions, xmin = x_start, xmax = x_end, color = GRAY2)
    
    # Add points (outliers) to these segments
    
    x_points = outliers
    y_points = [0] * len(outliers)
    
    # Draw points
    ax.scatter(x_points, y_points, s = 2, color = GRAY2)   # s - point size
    
    # set y-axis limits and labels
    ax.set_ylim([-1, 1])
    ax.set_yticks([0])
    ax.set_yticklabels([])
    ax.set_ylabel('Interquartile plot')
    
    return ax
    
def distribution_chart (df, station, month):
    '''
    built of different types of temperature distribution plots  
    df : DataFrame
    station, month : Station name and month of the data
    '''

    df = df[df['Station_name'] == station][month]

    fig, axes = plt.subplots(3,2, figsize  =(12, 6))
    # Add a global title for the entire shape
    fig.suptitle(
        f'Temperature distribution in {station} in {months_name[month]}', 
        fontsize = 22, color = GRAY4)

    # Customize the space between subplots
    fig.tight_layout(rect  =[0, 0, 1, 0.95])

    # Build plots for the column month and set Y-axis labels
    sns.boxplot(x = df, ax = axes[0, 0])
    axes[0, 0].set_ylabel('boxplot')
    sns.stripplot(x = df, ax = axes[2, 0])
    axes[2, 0].set_ylabel('Strip plot')  
    sns.kdeplot(x = df, fill = True, ax = axes[1, 0])
    axes[1, 0].set_yticklabels([])  
    IQR(df, ax = axes[0, 1])
    sns.swarmplot(x = df, ax = axes[2, 1])
    axes[2, 1].set_ylabel('Swarm plot')  
    sns.violinplot(x = df, ax = axes[1, 1])
    axes[1, 1].set_ylabel('Violin plot')  
    # Turn off the captions for the X-axes
    for ax in axes.flat:
        ax.set_xlabel('')  

    plt.show()

def time_serie(df, stations):
    '''
    linear plot of annual average temperature for the station
    df : DataFrame with data
    stations : list of the stations
    '''

    for station in stations:
        # Filter data by 'Station_name' category
        filtered_df = df[df['Station_name'] == station]
        # Draw a line graph
        plt.plot(filtered_df['Year'], filtered_df['Year_Mean'], marker = 'o')
    plt.xlabel('Year')
    plt.ylabel('Temperature')
    plt.title('Annual average temperature in Kharkiv and Lviv')
    
    plt.show()

def time_serie_upgrade(df, stations, color):
    '''
    linear plot of annual average temperature for the station
    df : DataFrame with data
    station : list of the stations
    color : colors of graphs
    '''

    # Configure plot font family to Arial
    with plt.rc_context({'font.family' : 'Arial'}):
        
        fig, ax = plt.subplots(figsize = (12, 6))

        for i, station in enumerate(stations):
            # Filter data by category 'Station_name'
            filtered_df = df[df['Station_name'] == station]
            # Build a line graph
            ax.plot(filtered_df['Year'], filtered_df['Year_Mean'], 
                    color = color[i])
             # Build a trend line - 
             # moving average with a window of 12 years
            filtered_df['Moving_Avg'] = \
                filtered_df['Year_Mean'].rolling(window = 12).mean()
            ax.plot(filtered_df['Year'], filtered_df['Moving_Avg'], 
                    color = GRAY7)

        plt.title('Annual average temperature in Kharkiv and Lviv', 
                  fontsize = 22, color = GRAY2)

        # Remove the graph frame
        plt.gca().spines[['top','right']].set_visible(False)
        # Change the color of the axis 
        plt.gca().spines[['bottom', 'left']].set_color(GRAY4)

        # Set the color of axis labels
        plt.tick_params(axis = 'both', colors = GRAY4)
        plt.tick_params(axis = 'x', length = 0)
        # Set the labels (positions) on the Y-axis
        ax.set_yticks([5, 8, 11])  
        # Set label captions
        ax.set_yticklabels(['5°C', '8°C', '11°C'], fontsize = 18)  
        
        # Label lines directly
        ax.text(2024, 10.3, 'Kharkiv', fontsize = 24, color = color[0])
        ax.text(2024, 9.6, 'Lviv', fontsize = 24, color = color[1])
        ax.text(2024, 8.6, 
                '*12-Year \nMoving Average \nof Annual \nTemperatures', 
                fontsize = 11, color = GRAY7)

        # Set the data limits for the y-axis and x-axis
        ax.set_xlim([1880, 2024])
        ax.set_ylim([4, 11])

        # Create a rectangle: (x, y, width, height)
        rect = patches.Rectangle((1989, 4), 35, 7, facecolor = GRAY9)

        # Add a rectangle to an axis
        ax.add_patch(rect)
        ax.text(1983.5, 10, '1989', fontsize = 24, color = GRAY4)
        # Add informative text
        ax.text(1990, 5, '$\\bf{increase}$ in average\n annual temperature',
                fontsize = 16, color = GRAY4)
      
        plt.show()

def monthly_temp(df, year, stations):
    '''
    Comparison of graphs of monthly average temperatures 
    in the city and in the region
    df : DataFrame with data
    stations : list of the stations
    year : year of observation
    '''

    for station in stations:

        # Filter data by categories 'Station_name', 'Year'
        filtered_row = df.loc[(df['Year'] == year) & 
                              (df['Station_name'] == station), months].T

        # Build a graph
        plt.plot(filtered_row.index, filtered_row.values, marker = 'o')
    
    plt.xlabel('Month')
    plt.ylabel('Temperature')
    plt.title(
     f'Monthly average temperature in {stations[0]} and {stations[0]} region in {year}' )
    
    plt.show()

def monthly_temp_upgrade(df, year, stations):
    '''
    Сomparison of graphs of monthly average temperatures 
    in the city and the region
    df : DataFrame with data
    stations : list of the stations
    year : year of observation
    '''

    # Colors for the graphs
    color = [ORANGE1] + [GRAY9] * (len(stations) - 1) + [BLUE2]

    # Filter data by categories 'Station_name', 'Year'
    filtered_df = df.loc[(df['Year'] == year) & 
                         (df['Station_name'].isin(stations)), months].T
    filtered_df.columns = stations

    # Calculate average temperature in the region
    filtered_df['Avg_region'] = \
        filtered_df[stations[1:]].mean(axis = 1).round(2)

    fig, ax = plt.subplots(figsize = (12, 6))

    # Building graphs
    for i,station in enumerate(stations + ['Avg_region']):
        ax.plot(filtered_df.index, filtered_df[station], color = color[i])

    plt.title(
        'Monthly average temperature in Kyiv and in the Kyiv region in 2015', 
        fontsize = '22', color = GRAY2)

    # Set the data limits for the y-axis and x-axis
    ax.set_xlim([0, 11])
    ax.set_ylim([-3, 24])

    # Remove the graph frame
    plt.gca().spines[['top','right']].set_visible(False)
    # Change the color of axis lines
    plt.gca().spines[['bottom', 'left']].set_color(GRAY4)

    # Setting the color of axis labels
    plt.tick_params(axis = 'both', colors = GRAY4)
  
    ax.set_yticks([0, 10, 20])  # Set the labels (positions) on the Y-axis
    ax.set_xticks([0, 3, 6, 9])  # Set the labels (positions) on the X-axis
    # Set label captions
    ax.set_yticklabels(['0°C', '10°C', '20°C'], fontsize = '18')     
    ax.set_xticklabels(['Jan', 'Apr', 'Jun', 'Oct'], fontsize = '18')     
        
    # Create a legend: rectangle (x, y, width, height)
    rect1 = patches.Rectangle((0.2, 22), 0.2, 1, facecolor = color[0])
    rect2 = patches.Rectangle((1.7, 22), 0.2, 1, 
                              facecolor = color[len(stations)])
    # Add rectangles to the axis
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    # Add rectangles to the axis
    ax.text(0.5, 22, 'Kyiv', fontsize = '14', color = GRAY2)
    ax.text(2, 22, 'Kyiv region', fontsize = '14', color = GRAY2)
  
    plt.show()

def heatmap(df, station):
    '''
    heatmap of average monthly temperature for the station
    df : DataFrame with data
    station : name of the station
    '''

    plt.title(f'Monthly average temperature in {station}')

    # Filter data by categories 'Station_name'
    filtered_df = df[df['Station_name'] == station]
    # Set 'Year' as index column
    filtered_df.set_index('Year', inplace = True)

    # Build heatmap
    sns.heatmap(filtered_df[months].T)

    plt.show()

def pseudocolor(ax, x, y, ymin, ymax, palette):
    '''
    plot with two-color segments
    ax : axes
    x, y : data for plotting
    ymin, ymax : boundary of the interval of the values
    palette : color palette
    '''

    l = (ymax - ymin) / PSEUDOMAP_N       # length of the interval
    for i in y.index:
        ax.plot([x[i], x[i]], [0, (y[i] - ymin) % l], 
                color = palette[int((y[i] - ymin) // l) + 1])
        ax.plot([x[i], x[i]], [(y[i] - ymin) % l, l], 
                color = palette[int((y[i] - ymin) // l)])
    return ax

def color_strips(ax, x, y, ymin, ymax, palette):
    '''
    Plot a color strip graph 
    ax : axes
    x, y : data to plot
    ymin, ymax : boundary of the interval of the values
    palette : color palette
    '''

    cmap = plt.get_cmap(palette)  
    norm = plt.Normalize(ymin, ymax)  
    
    # Build plot
    for i in y.index:
        ax.plot([x[i], x[i]], [0, 1], color = cmap(norm(y[i])))
    #print(cmap)
    ax.set_yticks([])  # Turn off y_ticks

    return ax

def annual_color_strips(df, station, palette):
    '''
    Plot a color strip graph of average annual temperature on the station 
    df : DataFrame
    station : name of the station
    palette : color palette
    '''
    filtered_df = df[df['Station_name'] == station]

    fig, ax = plt.subplots(figsize = (6, 1))
    fig.tight_layout(rect = [0, 0, 1, 1])
    ax = color_strips(ax, filtered_df['Year'], filtered_df['Year_Mean'], 
                      filtered_df['Year_Mean'].min(), 
                      filtered_df['Year_Mean'].max(), palette)
    plt.title(f'Annual average temperature in {station}')
    plt.show()

def plot_legend(type, ymin, ymax, ax, str, palette):
    '''
    build a legend of a plot 
    type : type of the plot 
    ymin, ymax : boundary of the interval of the values
    ax : axes 
    str : name of measurement units
    palette : color palette for color strips
    '''

    # dataframe of unit function y=x
    unit = pd.DataFrame(np.linspace(ymin, ymax, 101), columns = ['x'])
    # build a plot of the unit function by the type plotting
    ax = type(ax, unit['x'], unit['x'], ymin, ymax, palette)
    ax.set_xlim(ymin, ymax)
    ax.set_title(str, x = 1.05, y = -0.4)
    # turn off boundaries
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False) 
    ax.set_yticks([])           # turn off y-ticks
    return ax

def example(func, xmin = 0, xmax = 1, palette_pseudo = PSEUDOMAP_COLORS, 
            palette_strips = 'rainbow'):
    '''
    build example for comparison color strips and pseudocolor plots
    func : function to plot
    xmin, xmax : boundary of the interval to plot
    palette_pseudo : color array for pseudocolor
    palette_strips : color palette for color_strips
    '''

    # create a DataFrame of the function values
    F = pd.DataFrame(np.linspace(xmin, xmax, 501), columns = ['x'])
    F['y'] = func(F['x'])
    ymin = F['y'].min()
    ymax = F['y'].max()

    fig = plt.figure(figsize = (10,6))
    # create a grid for arranging graphs
    gs = gridspec.GridSpec(3, 1, height_ratios = [7, 1, 1])  
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Add a common title for the entire shape
    fig.suptitle('Conventional and two-tone pseudo coloring', fontsize = 22)
    # Customize the space between subplots
    fig.tight_layout(rect = [0, 0, 0.7, 1])

    ax1.set_xlim([xmin, xmax])
    ax2.set_xlim([xmin, xmax])
    ax3.set_xlim([xmin, xmax])

    # plot the graphs
    ax1.plot(F['x'], F['y'])
    ax2 = color_strips(ax2, F['x'], F['y'], ymin, ymax, palette_strips)
    ax3 = pseudocolor(ax3, F['x'], F['y'], ymin, ymax, palette_pseudo)
    # turn off boundaries
    ax2.spines[['top', 'right', 'bottom', 'left']].set_visible(False) 
    ax3.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax2.set_yticks([])           # turn off y-ticks
    ax3.set_yticks([])    

    # Add a legend right to a large matrix of subplots
    # Location of the legend for pseudocolor
    ax_legend_pseudo = fig.add_axes([0.75, 0.08, 0.15, 0.03])  
    plot_legend(pseudocolor, ymin, ymax, 
                ax_legend_pseudo, '', palette_pseudo)
    # Location of the legend for color strips
    ax_legend_strips = fig.add_axes([0.75, 0.23, 0.15, 0.03])  
    plot_legend(color_strips, ymin, ymax, 
                ax_legend_strips, '', palette_strips)
 

    plt.show()

def graph_pseudocolor(df, stations, ymin, ymax, 
                      palette_pseudo = PSEUDOMAP_COLORS, 
                      palette_strips = 'rainbow'):
    '''
    plot one-dimentional pseudocoloring graph
    value_x, value_y : coordinates of the graph
    ymin, ymax : boundary of the colormap
    palette_pseudo : color array for pseudocolor
    palette_strips : color palette for color_strips
    '''

    l = (ymax - ymin) / PSEUDOMAP_N       # interval length

    fig, ax = plt.subplots(len(stations), 3, sharex = True, figsize = (12,6))
    # Add a common title for the entire shape
    fig.suptitle('Annual average temperatures on different stations', 
                 fontsize = 22, color = GRAY4)
    # Customize the space between subplots
    fig.tight_layout(rect = [0.08, 0, 1, 0.9])

    # build plots
    for j, station in enumerate(stations):
        ax[j][0].set_xlim([1880, 2024])
        ax[j][0].set_ylim([0,l])
        ax[j][2].set_xlim([1880, 2024])
        ax[j][2].set_ylim([5, 15])

        filtered_df = df[df['Station_name'] == station]
        x = filtered_df['Year']
        y = filtered_df['Year_Mean']
        # build pseudocolor plots
        ax[j][0] = pseudocolor(ax[j][0], x, y, ymin, ymax, palette_pseudo)
        # build color stripes
        ax[j][1] = color_strips(ax[j][1], x, y, ymin, ymax, palette_strips)
        # build linear plots
        ax[j][2].plot(x,y)

        ax[j][0].text(1840, 0.3, station, fontsize = '12', color = GRAY4)
        ax[j][0].set_yticks([])
        ax[j][2].set_yticks([10])
        ax[j][2].set_yticklabels(['10°C'])


    # Add a legend over a large matrix of subplots
    # Location of the legend for pseudocolor
    ax_legend_pseudo = fig.add_axes([0.12, 0.85, 0.15, 0.03])  
    plot_legend(pseudocolor, ymin, ymax, 
                ax_legend_pseudo, '°C', palette_pseudo)
    # Location of the legend for color strips
    ax_legend_strips = fig.add_axes([0.42, 0.85, 0.15, 0.03])  
    plot_legend(color_strips, ymin, ymax, 
                ax_legend_strips, '°C', palette_strips)
    
    plt.show()

def geomap (df) :
    '''
    Interactive map of Ukraine 
    with the annual average temperature by observation stations, 
    the year is selected using the slider
    df : Dataframe with data
    '''

    # Plot the map of Ukraine with the stations
    fig = go.Figure( layout = go.Layout(width = 800, height = 600),
        data = go.Choropleth(
        locations = ["UKR"],
        locationmode = 'ISO-3',
        z = [1],      # Value for the country color
        zmin = 0,
        zmax = 100,
        showscale = False,
        marker_line_color = 'black',
        colorscale = "Blues",
        hoverinfo = 'none'
        )
    )
   
    # Add data for each year to the graph
    years = sorted(df['Year'].unique())
    for year in years:
        year_df = df[df['Year'] == year]
        
        # Add datapoints to the map
        fig.add_trace(go.Scattergeo(
            lon = year_df['Lon'],   # longitude of the station
            lat = year_df['Lat'],   # latitude of the station
            text = year_df['Station_name'] + '\n T=' + \
                 year_df ['Year_Mean'].astype(str) + '°C',
            hoverinfo = 'lon+lat+text',   #pop-up tips
            showlegend = False,
            mode = 'markers',
            name = str(year),
            marker = dict(
                symbol = 'circle',
                size = 8,
                # Annual average temperature on the station
                color = year_df['Year_Mean'],   
                colorscale = 'ylorrd',    
                cmin = 5,
                cmax = 15,
                colorbar = dict(      
                    tickvals = [5, 10, 15],
                    ticktext = ['5°C', '10°C', '15°C'],
                    thickness = 15,
                    y = 0.5
                    ),  
                showscale = True  
            ),
            # Initially visible only in the first year
            visible=(year == years[0])  
        ))

    # Customize slider steps
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method = 'update',
            label = str(year),
            # Turn off all traces
            args = [{'visible': [False] * (len(years) + 1)}] 
        )
        # Make the trace for the map visible
        step['args'][0]['visible'][0] = True    
        # Make the trace for the current year visible
        step['args'][0]['visible'][i+1] = True  
        steps.append(step)

    # Customize the map
    fig.update_geos(
        visible = True, 
        resolution = 50,
        showcountries = False, 
        lataxis_range = [44, 53], 
        lonaxis_range = [22, 41],
        projection_type = 'natural earth'
    )
    
    fig.update_layout(
        title = {
            'text':"Annual average temperature in Ukraine",
            'x': 0.5,               # center alignment
            'xanchor': 'center',    # Anchor for horizontal location
        },
        sliders = [{
            'active': 0,
            'currentvalue': {
                'prefix': 'Year: ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10},
            'steps': steps
        }]
    )

    fig.show()
  
# select data for Ukraine
ukr_data = select_data('v4.mean_GISS_homogenized.txt', 'UP')
print(ukr_data.head())
print(ukr_data.info())
ukr_data.to_csv('ukr_data.csv', index = False)


# convert to numerical values
ukr_data = preprocess(ukr_data)
print(ukr_data.head())
print(ukr_data.info())

# discover history of temperature stations
count_df = history(ukr_data, 1880, 2024, 20)
history_grouped(count_df)

# Group data by ID station and replace outliers
ukr_data = ukr_data.groupby('ID', group_keys = False).apply(replace_outliers)

# calculate mean temperatures of the year
ukr_data['Year_Mean'] = ukr_data[months].mean(axis = 1).round(2)
print (ukr_data.head())
print (ukr_data.info())

# save processed data
ukr_data.to_csv('ukr_processed.csv', index = False)

# discover different types of distribution presentations
distribution_chart(ukr_data, 'KHARKIV', 'FEB')

# discover mean temperature of the year in some stations
time_serie(ukr_data, ['KHARKIV', 'LVIV'])
time_serie_upgrade(ukr_data, ['KHARKIV', 'LVIV'], color = [BLUE2,ORANGE1])

# discover monthly temperature in the city and in the region
monthly_temp(ukr_data, 2015, ['KIEV', 'BORYSPIL', 'YAHOTYN', 'MYRONIVKA'])
monthly_temp_upgrade(ukr_data, 2015, 
                     ['KIEV', 'BORYSPIL', 'YAHOTYN', 'MYRONIVKA'])

# plot a heat map of temperature on the station
heatmap(ukr_data, 'KHARKIV')

#monthly_color_strips(ukr_data, 'KHARKIV', 1989, 2024, 'coolwarm')
annual_color_strips (ukr_data, 'KHARKIV', 'coolwarm')

# discover pseudocolor presentation of temperature
example(lambda x: np.sin(np.sqrt(x) * np.pi), xmin = 0, xmax = 1)
graph_pseudocolor(ukr_data, 
                  ['KIEV','LVIV','POLTAVA','KHARKIV','LUGANSK','UMAN',\
                   'ODESA', 'NIKOLAEV','KERCH','YALTA'], ymin = 4, ymax = 16)

# plot interactive geomap of the temperatures
geomap(ukr_data)



