from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Read the CSV file into a DataFrame
df = pd.read_csv('filtered_count_df.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    reg_number = request.form['reg_number']
    # Filter the DataFrame to find matching advertisement numbers for the input registration number
    matching_ads_data = df[df['Reg No 1'] == int(reg_number)].values
    matching_ads = []
    for row in matching_ads_data:
        matching_ads.append({
            'Ad Num 1': row[0],
            'Ad Num 2': row[1],
            'Count': row[4]
        })
    return render_template('result.html', reg_number=reg_number, matching_ads=matching_ads)

if __name__ == '__main__':
    app.run(debug=True)
