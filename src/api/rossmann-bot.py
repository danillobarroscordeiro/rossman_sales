import pandas as pd
import json
import requests
from flask import Flask, request, Response
import waitress

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TOKEN = '6146197280:AAGXqdrqL1dlK4035faLC9fMggQJf9-0fJ4'
# 'https://api.telegram.org/botTOKEN/getMe'

# 'https://api.telegram.org/botTOKEN/getUpdates'

# 'https://api.telegram.org/botTOKEN/sendMessage?chat_id=1325084193&text=Hi Meigarom'

# 'https://api.telegram.org/botTOKEN/setWebhook?url=ec2-54-159-137-138.compute-1.amazonaws.com



def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    url_request = requests.post(url, json={'text': text})
    print('Status Code {}').format(url_request.status_code)

    logger.debug('Status Code: %d', url_request.status_code)


    return None

def load_dataset(store_id):

    df10 = pd.read_csv(
        '/home/ubuntu/rossman_sales/data/raw/test.csv'
        )
    df_store_raw = pd.read_csv(
        '/home/ubuntu/rossman_sales/data/raw/store.csv'
        )

    #merge test dataset + store
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    #choose store from prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:
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
    url = 'http://ec2-54-159-137-138.compute-1.amazonaws.com:5000/predict'
    header = {'Content-type':'application/json'}
    df = data

    logger.debug('Status Code: %d', request_api.status_code)


    request_api = requests.post(url, data=df, headers=header)
    print('Status Code {}'.format(request_api.status_code))

    df_result = pd.DataFrame(
        request_api.json(), columns=request_api.json()[0].keys()
        )

    return df_result



def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'
    
    return chat_id, store_id

app = Flask(__name__)
@app.route('/bot', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            data = load_dataset(store_id)
            if data != 'error':
                df_result = predict(data)
                df_result_final = (
                    df_result[['store', 'prediction']].groupby('store')
                    .sum()
                    .reset_index()         
                )
                msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
                df_result_final.loc['store'].values[0],
                df_result_final.loc['prediction'].values[0])

                send_message(chat_id, msg)
                return Response('Ok', status=200)
                
            else:
                send_message(chat_id, 'Store not avaliable')
                return Response('Ok', status=200)
        else:
                send_message(chat_id, 'Store not avaliable')
                return Response('Ok', status=200)
            
    else:
        return '<h1> Rossman Telegram BOT <h1>'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
