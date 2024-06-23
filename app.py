import pandas as pd
import streamlit as st
import os
import pickle
import altair as alt

# import data with GT in it
files = os.listdir('data')
files = [file for file in files if 'GT' in file]

data = pd.DataFrame()

for file in files:
    data = pd.concat([data, pd.read_csv(f'data/{file}')])

data = data[(data['TaggedPitchType'].isin(['Dropball', 'Fastball', 'Riseball', 'Changeup', 'Curveball', 'Screwball'])) & (data['PitcherTeam'] == 'Georgia tech')]

ab_scores = pd.read_csv('ab_scores.csv')

data = pd.merge(data, ab_scores, left_on='PitchCall', right_on='Feature', how='inner')

scores_by_pitcher = data.groupby(['Pitcher', 'TaggedPitchType'])['Score'].agg(['mean', 'count']).reset_index()
scores_by_pitcher = scores_by_pitcher[scores_by_pitcher['count'] > 10]

worst_score = scores_by_pitcher['mean'].min()
best_score = scores_by_pitcher['mean'].max()

pitch_call_machine_tab, pitcher_analysis_tab = st.tabs(['Pitch Call Machine', 'Pitcher Analysis'])

with pitch_call_machine_tab:

    st.title('Pitch Call Machine')

    pitcher = st.selectbox('Pitcher', data['Pitcher'].unique())

    model = pickle.load(open('models/' + pitcher + '.pkl', 'rb'))

    with open('models/' + pitcher + ' Features.txt', 'r') as file:
        features = [line.strip() for line in file.readlines()]

    pitcher_data = data[data['Pitcher'] == pitcher]

    batter_hand = st.selectbox('Batter Hand', ['Right', 'Left'])
    balls = st.number_input('Balls', 0, 3, 0)
    strikes = st.number_input('Strikes', 0, 2, 0)
    previous_pitch = st.selectbox('Previous Pitch', ['NA'] + list(pitcher_data['TaggedPitchType'].unique()))

    pitch = st.button('Next Pitch Call')

    if pitch:
        pitch_weights = {'Fastball': 0, 'Changeup': 0, 'Curveball': 0, 'Riseball': 0, 'Dropball': 0, 'Screwball': 0}

        # List of all possible features based on the initial model training
        pitch_types = ['Fastball', 'Changeup', 'Curveball', 'Riseball', 'Dropball', 'Screwball']
        tagged_pitch_columns = [f'TaggedPitchType_{pitch}' for pitch in pitch_types]
        interaction_columns = [f'TaggedPitchType_{t}_LagPitchType_{l}' for t in pitch_types for l in pitch_types]

        required_columns = tagged_pitch_columns + \
                        [f'BatterSide_Left', f'BatterSide_Right'] + \
                        [f'Balls'] + \
                        [f'Strikes'] + \
                        [f'LagPitchType_{pitch}' for pitch in pitch_types] + \
                        interaction_columns

        for i in pd.unique(pitcher_data['TaggedPitchType']):
            # Initialize the DataFrame with the required columns filled with zeros
            pitcher_data = pd.DataFrame(columns=required_columns)
            pitcher_data.loc[0] = 0

            # Set the one-hot encoded values for the input features
            pitcher_data.at[0, f'BatterSide_{batter_hand}'] = 1
            pitcher_data.at[0, f'Balls'] = balls
            pitcher_data.at[0, f'Strikes'] = strikes
            pitcher_data.at[0, f'LagPitchType_{previous_pitch}'] = 1
            pitcher_data.at[0, f'TaggedPitchType_{i}'] = 1

            # Set the interaction terms
            for pitch_type in pitch_types:
                interaction_term = f'TaggedPitchType_{i}_LagPitchType_{pitch_type}'
                if interaction_term in pitcher_data.columns:
                    pitcher_data.at[0, interaction_term] = int(previous_pitch == pitch_type)

            # Ensure all required columns are present
            for col in required_columns:
                if col not in pitcher_data.columns:
                    pitcher_data[col] = 0

            pitcher_data = pitcher_data[features]
            print(pitcher_data)

            # Make the prediction
            prediction = model.predict(pitcher_data)
            pitch_weights[i] = prediction
        
        weights_df = pd.DataFrame(pitch_weights, index=[0]).T.sort_values(by=0, ascending=False).rename(columns={0: 'Value'})
        weights_df = weights_df[weights_df['Value'] > 0]

        st.markdown(f'''
                    <p style="font-size: 40px; font-weight: bold; text-align: center">1. {weights_df.index[0]}</p>
                    <p style="font-size: 30px; font-weight: bold; text-align: center">2. {weights_df.index[1]}</p>
                    ''', unsafe_allow_html=True)

with pitcher_analysis_tab:

    pitcher = st.selectbox('Pitcher', data['Pitcher'].unique(), key='pitcher_analysis_pitcher')

    analysis_data = data[data['Pitcher'] == pitcher]

    # get average Value by pitch type
    analysis_data['Score'] = analysis_data['Score'].astype(float)
    avg_value = analysis_data.groupby('TaggedPitchType')['Score'].agg(['mean', 'count'])

    avg_value = avg_value[(avg_value['count'] > 10)]

    avg_value = avg_value['mean']

    # normalize values by the worst and best score to get scale from 0 to 10
    avg_value = round(((avg_value - worst_score) / (best_score - worst_score)) * 10, 2)

    if len(avg_value) > 0:

        st.write()
        st.markdown(f'<h1 style="text-align: center">Pitch Scores</h1>', unsafe_allow_html=True)
        chart = alt.Chart(avg_value.reset_index()).mark_bar().encode(
            x=alt.X('TaggedPitchType', title='Pitch Type'),
            y=alt.Y('mean', title='Score', scale=alt.Scale(domain=[0, 10])),
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart)
    else:
        st.write('Not enough data available for this pitcher')