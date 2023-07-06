import pandas as pd
import json
import requests
from flask import Flask, request, Response
#loading test dataset

TOKEN = '6367271902:AAEYimkELSwkGrIUg-1D9SgxLlhNypLWjxQ'
'https://api.telegram.org/bot6367271902:AAEYimkELSwkGrIUg-1D9SgxLlhNypLWjxQ/getMe'

'https://api.telegram.org/bot6367271902:AAEYimkELSwkGrIUg-1D9SgxLlhNypLWjxQ/getUpdates'

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    url_request = requests.post(url, json={'text': text})
    print('Status Code {}').format(url_request.status_code)

    return None
'https://api.telegram.org/bot6367271902:AAEYimkELSwkGrIUg-1D9SgxLlhNypLWjxQ/sendMessage?chat_id=1325084193&text=Hi Meigarom'

def load_dataset(store_id):

    df10 = pd.read_csv(
        '/home/dbcordeiro@sefaz.al.gov.br/Documents/repos/rossman_sales/data/test.csv'
        )
    df_store_raw = pd.read_csv(
        '/home/dbcordeiro@sefaz.al.gov.br/Documents/repos/rossman_sales/data/store.csv'
        )

    #merge test dataset + store
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    #choose store from prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty():
        #remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis=1)

        #convert Dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))
    
    else:
        data = 'error'

    return data

def predict(data):
    #API Call
    url = 'http://0.0.0.0:5000/rossmann/predict'
    header = {'Content-type':'application/json'}
    data = data

    request_api = requests.post(url, data=data, headers=header)
    print('Status Code {}'.format(request_api.status_code))

    df_result = pd.DataFrame(
        request_api.json(), columns=request_api.json()[0].keys()
        )

    return df_result

# df_result_final = (
#     df_result[['store', 'prediction']].groupby('store')
#     .sum()
#     .reset_index()         
# )

# for i in range(len(df_result_final)):
#     print(
#         'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
#         df_result_final.loc[i, 'store'],
#         df_result_final.loc[i, 'prediction']
#         )
#     )

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method = 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            data = load_dataset(store_id)
            if data != 'error':
                d1 = predict(data)
            else:
                send_message(chat_id, 'Store not avaliable')
                return Response('Ok', status=200)

    else:
        send_message(chat_id, 'There is no store ID avaliable')
        return Response('Ok', status=200)


if __name__ = '__main__':
    app.run(host='0.0.0.0', port=5000)