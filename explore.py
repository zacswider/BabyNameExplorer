import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.gridspec import GridSpec
import urllib.request
import os


# Global state
state = {
    'slider_value': 0,
    'sex_filter': 'All',
    'df': None,
    'df_2015_male': None,
    'df_2015_female': None,
    'df_2015_all': None,
    'ax_plot': None,
    'ax_text': None,
    'fig': None,
    'slider': None,
    'btn_back': None,
    'btn_forward': None,
    'text_box': None,
    'threshold_box': None,
    'num_names_box': None,
    'search_mode': False,  # Track if we're showing a searched name
    'threshold': 0,  # Minimum peak popularity threshold
    'num_names_to_show': 5,  # Number of names to display
}


def ensure_data_file():
    """Download the CSV file from GitHub if it doesn't exist locally."""
    csv_filename = 'babyNamesUSYOB-full.csv'

    if not os.path.exists(csv_filename):
        print(f"Data file not found locally. Downloading from GitHub...")
        # Use media.githubusercontent.com for LFS files
        url = 'https://media.githubusercontent.com/media/zacswider/BabyNameExplorer/main/babyNamesUSYOB-full.csv'
        try:
            urllib.request.urlretrieve(url, csv_filename)
            print(f"Successfully downloaded {csv_filename}")
        except Exception as e:
            print(f"Error downloading data file: {e}")
            raise


def load_and_prepare_data():
    """Load CSV and create pre-ranked 2015 DataFrames for each sex filter."""
    ensure_data_file()
    print("Loading data...")
    df = pd.read_csv('babyNamesUSYOB-full.csv')

    # Compute max count ever for each name-sex combination
    print("Computing peak popularity for each name...")
    max_counts = df.groupby(['Name', 'Sex'])['Number'].max().reset_index()
    max_counts.columns = ['Name', 'Sex', 'MaxCountEver']

    # Filter to 2015 data for ranking
    df_2015 = df[df['YearOfBirth'] == 2015].copy()

    # Merge max counts into 2015 data
    df_2015 = df_2015.merge(max_counts, on=['Name', 'Sex'], how='left')

    # Create ranked DataFrames for each sex
    df_2015_male = df_2015[df_2015['Sex'] == 'M'].sort_values('Number', ascending=True).reset_index(drop=True)
    df_2015_female = df_2015[df_2015['Sex'] == 'F'].sort_values('Number', ascending=True).reset_index(drop=True)
    df_2015_all = df_2015.sort_values('Number', ascending=True).reset_index(drop=True)

    print(f"Loaded {len(df):,} total records")
    print(f"2015 data: {len(df_2015_male):,} male names, {len(df_2015_female):,} female names")

    return df, df_2015_male, df_2015_female, df_2015_all


def get_names_at_percentile(slider_value, sex_filter):
    """Convert slider position (0-100) to names at that percentile.

    Slider 0 = least popular
    Slider 100 = most popular
    """
    # Select appropriate ranked DataFrame
    if sex_filter == 'Male':
        df_ranked = state['df_2015_male']
    elif sex_filter == 'Female':
        df_ranked = state['df_2015_female']
    else:
        df_ranked = state['df_2015_all']

    # Apply threshold filter
    threshold = state['threshold']
    if threshold > 0:
        df_ranked = df_ranked[df_ranked['MaxCountEver'] >= threshold].reset_index(drop=True)

    total_names = len(df_ranked)
    num_names = state['num_names_to_show']

    # Ensure we have at least num_names
    if total_names < num_names:
        # Return what we have, even if less than requested
        selected = df_ranked[['Name', 'Sex', 'Number']].copy()
    else:
        # Convert slider (0-100) to starting rank position
        # slider=0 -> start at rank 0 (least popular, since sorted ascending)
        # slider=100 -> start at rank total_names-num_names (most popular)
        percentile = slider_value / 100.0
        start_rank = int(percentile * (total_names - num_names))
        start_rank = max(0, min(start_rank, total_names - num_names))

        # Get num_names starting from this rank
        selected = df_ranked.iloc[start_rank:start_rank + num_names][['Name', 'Sex', 'Number']].copy()

    names = selected['Name'].tolist()
    sexes = selected['Sex'].tolist()
    counts = selected['Number'].tolist()

    return names, sexes, counts


def get_time_series_data(names, sexes):
    """Extract time series data for each name.

    Returns dict mapping name -> (years_array, counts_array)
    """
    df = state['df']

    # Get year range from data
    min_year = df['YearOfBirth'].min()
    max_year = df['YearOfBirth'].max()
    all_years = np.arange(min_year, max_year + 1)

    result = {}
    for name, sex in zip(names, sexes):
        # Filter by name and sex
        name_data = df[(df['Name'] == name) & (df['Sex'] == sex)]

        # Create a complete time series with missing years as 0
        year_counts = name_data.set_index('YearOfBirth')['Number'].reindex(all_years, fill_value=0)

        # Create display name with sex indicator for disambiguation
        display_name = f"{name} ({sex})"
        result[display_name] = (all_years, year_counts.values)

    return result


def update_plot(names_data):
    """Redraw the line plot with new time series data."""
    ax = state['ax_plot']
    ax.clear()

    # Plot each name's time series
    for display_name, (years, counts) in names_data.items():
        ax.plot(years, counts, label=display_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Babies', fontsize=11)
    ax.set_title('Baby Name Popularity Over Time (1880-2015)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set x-axis limits
    ax.set_xlim(1880, 2015)


def update_name_list(names, sexes, counts):
    """Update the right panel with the list of selected names."""
    ax = state['ax_text']
    ax.clear()
    ax.axis('off')

    # Calculate min and max counts
    min_count = min(counts)
    max_count = max(counts)

    # Build text content
    lines = ["Selected Names\n(2015 rank)\n"]
    lines.append(f"count: {min_count:,}-{max_count:,}\n")
    lines.append("-" * 20 + "\n\n")

    for i, (name, sex) in enumerate(zip(names, sexes), 1):
        lines.append(f"{i}. {name} ({sex})\n")

    text_content = "".join(lines)
    ax.text(0.05, 0.95, text_content, fontsize=11,
            verticalalignment='top', family='monospace',
            transform=ax.transAxes)


def update_display():
    """Update both the plot and name list based on current state."""
    # Exit search mode when using slider/buttons
    state['search_mode'] = False

    names, sexes, counts = get_names_at_percentile(state['slider_value'], state['sex_filter'])
    names_data = get_time_series_data(names, sexes)
    update_plot(names_data)
    update_name_list(names, sexes, counts)
    state['fig'].canvas.draw_idle()


def on_slider_change(val):
    """Callback for slider value changes."""
    state['slider_value'] = val
    update_display()


def on_radio_change(label):
    """Callback for radio button selection changes."""
    state['sex_filter'] = label
    # Also exit search mode and update display
    state['search_mode'] = False
    update_display()


def on_back_button(event):
    """Callback for back button - show previous N names."""
    slider = state['slider']
    if slider is None:
        return

    # Get filtered dataframe based on current sex filter and threshold
    if state['sex_filter'] == 'Male':
        df_ranked = state['df_2015_male']
    elif state['sex_filter'] == 'Female':
        df_ranked = state['df_2015_female']
    else:
        df_ranked = state['df_2015_all']

    # Apply threshold filter to match what get_names_at_percentile sees
    threshold = state['threshold']
    if threshold > 0:
        df_ranked = df_ranked[df_ranked['MaxCountEver'] >= threshold].reset_index(drop=True)

    total_names = len(df_ranked)
    num_names = state['num_names_to_show']

    if total_names <= num_names:
        return  # Can't navigate if not enough names

    # Calculate step to move by exactly num_names non-overlapping names
    # step in slider units = (num_names / (total_names - num_names)) * 100
    step = (num_names / (total_names - num_names)) * 100

    new_value = max(0, slider.val - step)
    slider.set_val(new_value)


def on_forward_button(event):
    """Callback for forward button - show next N names."""
    slider = state['slider']
    if slider is None:
        return

    # Get filtered dataframe based on current sex filter and threshold
    if state['sex_filter'] == 'Male':
        df_ranked = state['df_2015_male']
    elif state['sex_filter'] == 'Female':
        df_ranked = state['df_2015_female']
    else:
        df_ranked = state['df_2015_all']

    # Apply threshold filter to match what get_names_at_percentile sees
    threshold = state['threshold']
    if threshold > 0:
        df_ranked = df_ranked[df_ranked['MaxCountEver'] >= threshold].reset_index(drop=True)

    total_names = len(df_ranked)
    num_names = state['num_names_to_show']

    if total_names <= num_names:
        return  # Can't navigate if not enough names

    # Calculate step to move by exactly num_names non-overlapping names
    step = (num_names / (total_names - num_names)) * 100

    new_value = min(100, slider.val + step)
    slider.set_val(new_value)


def on_text_submit(text):
    """Callback for text box submission - search for a name and display it."""
    if not text.strip():
        # If empty, return to slider mode
        state['search_mode'] = False
        update_display()
        return

    name = text.strip()
    df = state['df']

    # Search for the name based on current sex filter
    if state['sex_filter'] == 'Male':
        matches = df[(df['Name'].str.lower() == name.lower()) & (df['Sex'] == 'M')]
    elif state['sex_filter'] == 'Female':
        matches = df[(df['Name'].str.lower() == name.lower()) & (df['Sex'] == 'F')]
    else:  # 'All'
        matches = df[df['Name'].str.lower() == name.lower()]

    if matches.empty:
        # Name not found, do nothing
        return

    # Enter search mode
    state['search_mode'] = True

    # Get unique name-sex combinations
    name_sex_pairs = matches[['Name', 'Sex']].drop_duplicates()

    names = []
    sexes = []
    counts_2015 = []

    for _, row in name_sex_pairs.iterrows():
        names.append(row['Name'])
        sexes.append(row['Sex'])

        # Get 2015 count for this name-sex combination
        match_2015 = df[(df['Name'] == row['Name']) &
                        (df['Sex'] == row['Sex']) &
                        (df['YearOfBirth'] == 2015)]
        if not match_2015.empty:
            counts_2015.append(match_2015['Number'].iloc[0])
        else:
            counts_2015.append(0)

    # Get time series data and update display
    names_data = get_time_series_data(names, sexes)
    update_plot(names_data)
    update_name_list(names, sexes, counts_2015)
    state['fig'].canvas.draw_idle()


def on_threshold_submit(text):
    """Callback for threshold box submission - filter names by peak popularity."""
    if not text.strip():
        state['threshold'] = 0
    else:
        try:
            threshold = int(text)
            state['threshold'] = max(0, threshold)  # Ensure non-negative
        except ValueError:
            # Invalid input, ignore
            return

    # Update display with new threshold
    if not state['search_mode']:
        update_display()


def on_num_names_submit(text):
    """Callback for num names box submission - change number of names to show."""
    if not text.strip():
        state['num_names_to_show'] = 5  # Default
    else:
        try:
            num_names = int(text)
            state['num_names_to_show'] = max(1, num_names)  # Ensure at least 1
        except ValueError:
            # Invalid input, ignore
            return

    # Update display with new number of names
    if not state['search_mode']:
        update_display()


def setup_gui():
    """Initialize the matplotlib figure with all widgets and layout."""
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Baby Names Explorer', fontsize=14, fontweight='bold')
    state['fig'] = fig

    # Create GridSpec for main layout
    gs = GridSpec(1, 4, figure=fig, left=0.08, right=0.95, top=0.75, bottom=0.1,
                  wspace=0.3)

    # Create axes for plot (left 3/4)
    ax_plot = fig.add_subplot(gs[0, 0:3])
    state['ax_plot'] = ax_plot

    # Create axes for text display (right 1/4)
    ax_text = fig.add_subplot(gs[0, 3])
    ax_text.axis('off')
    state['ax_text'] = ax_text

    # Create back button (left side of slider)
    ax_back = fig.add_axes([0.05, 0.88, 0.05, 0.03])
    btn_back = Button(ax_back, '◄ Back', color='lightgray', hovercolor='gray')
    btn_back.label.set_fontsize(9)

    # Create slider at top (between buttons)
    ax_slider = fig.add_axes([0.15, 0.88, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Popularity',
        valmin=0,
        valmax=100,
        valinit=0,
        valstep=1,
    )
    slider.label.set_fontsize(11)
    slider.valtext.set_fontsize(10)

    # Create forward button (right side of slider)
    ax_forward = fig.add_axes([0.88, 0.88, 0.05, 0.03])
    btn_forward = Button(ax_forward, 'Next ►', color='lightgray', hovercolor='gray')
    btn_forward.label.set_fontsize(9)

    # Add labels for slider endpoints
    fig.text(0.11, 0.885, 'Least\nPopular', fontsize=9, ha='center', va='center')
    fig.text(0.86, 0.885, 'Most\nPopular', fontsize=9, ha='center', va='center')

    # Create radio buttons for sex filter
    ax_radio = fig.add_axes([0.4, 0.78, 0.2, 0.08])
    radio = RadioButtons(ax_radio, ('Male', 'Female', 'All'), active=2)
    for label in radio.labels:
        label.set_fontsize(10)

    # Create text box for name search
    ax_textbox = fig.add_axes([0.35, 0.70, 0.3, 0.04])
    text_box = TextBox(ax_textbox, 'Search Name:', initial='', label_pad=0.01)
    text_box.label.set_fontsize(10)

    # Create text box for number of names to show
    ax_num_names = fig.add_axes([0.35, 0.63, 0.3, 0.04])
    num_names_box = TextBox(ax_num_names, 'Num Names:', initial='5', label_pad=0.01)
    num_names_box.label.set_fontsize(10)

    # Create text box for threshold filter
    ax_threshold = fig.add_axes([0.35, 0.56, 0.3, 0.04])
    threshold_box = TextBox(ax_threshold, 'Min Peak Count:', initial='0', label_pad=0.01)
    threshold_box.label.set_fontsize(10)

    # Store widgets in state to prevent garbage collection
    state['slider'] = slider
    state['btn_back'] = btn_back
    state['btn_forward'] = btn_forward
    state['text_box'] = text_box
    state['num_names_box'] = num_names_box
    state['threshold_box'] = threshold_box

    # Connect callbacks
    slider.on_changed(on_slider_change)
    radio.on_clicked(on_radio_change)
    btn_back.on_clicked(on_back_button)
    btn_forward.on_clicked(on_forward_button)
    text_box.on_submit(on_text_submit)
    num_names_box.on_submit(on_num_names_submit)
    threshold_box.on_submit(on_threshold_submit)

    return fig, slider, radio


def main():
    """Main entry point for the baby names explorer application."""
    # Load and prepare data
    df, df_2015_male, df_2015_female, df_2015_all = load_and_prepare_data()

    # Store in global state
    state['df'] = df
    state['df_2015_male'] = df_2015_male
    state['df_2015_female'] = df_2015_female
    state['df_2015_all'] = df_2015_all

    # Setup GUI
    fig, slider, radio = setup_gui()

    # Initial display (slider at 0 = least popular, All names)
    update_display()

    # Show the GUI
    plt.show()


if __name__ == "__main__":
    main()
