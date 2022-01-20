from typing import Union
import pandas as pd
import requests
import io

def nasdaq_get_history(nasdaq_id: str, from_date: str, to_date: str, columns_to_add: Union[dict[str, str], bool]=False ) -> Union[pd.DataFrame, str]:
    """Does a request to the nasdaq api and returns dataframe of historical data for requested nasdaq_id/company and date-interval.

    Args:
        nasdaq_id (str): the company's nasdaq identifier, needed for the request.
        from_date (str): history start-date. Y-m-d
        to_date (str): history end-date. Y-m-d
        columns_to_add (dict | bool): Defaults to False. If not false, provide a dict with keys as as column names and value as column value.
    Returns:
        pd.DataFrame: Datframe with historical data of the company. If request fails a string-error-message will be sent.
        columns: 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    """
    header = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'Origin': 'http://www.nasdaqomxnordic.com',
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Referer': f'http://www.nasdaqomxnordic.com/aktier/microsite?Instrument={nasdaq_id}',
        'Accept-Language': 'sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    xmlquery = f'''<post>
    <param name="Exchange" value="NMF"/>
    <param name="SubSystem" value="History"/>
    <param name="Action" value="GetDataSeries"/>
    <param name="AppendIntraDay" value="no"/>
    <param name="Instrument" value="{nasdaq_id}"/>
    <param name="FromDate" value="{from_date}"/>
    <param name="ToDate" value="{to_date}"/>
    <param name="hi__a" value="0,5,6,3,1,2,4,21,8,10,12,9,11"/>
    <param name="ext_xslt" value="/nordicV3/hi_csv.xsl"/>
    <param name="OmitNoTrade" value="true"/>
    <param name="ext_xslt_lang" value="en"/>
    <param name="ext_xslt_options" value=",adjusted,"/>
    <param name="ext_contenttype" value="application/ms-excel"/>
    <param name="ext_xslt_hiddenattrs" value=",iv,ip,"/>
    <param name="ext_xslt_tableId" value="historicalTable"/>
    <param name="DefaultDecimals" value="false"/>
    <param name="app" value="/aktier/microsite"/>
    </post>">'''
    response = requests.post('http://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx', headers=header, data={'xmlquery': xmlquery}, verify=False, timeout=20)
    if len(response.content) < 80:
        return f'Request failed, please try again. Error response: {response.text}'

    csv_text = response.content.decode('utf-8')[7:].replace('\r\n', '\n')
    df = pd.read_csv(io.StringIO(csv_text), sep=';', decimal=',', parse_dates=['Date'])\
            .rename(columns={'Date':'date',
                            'High price':'high',
                            'Low price':'low',
                            'Opening price':'open',
                            'Closing price':'close',
                            'Bid':'bid',
                            'Ask':'ask',
                            'Total volume':'volume',
                            'Trades':'trades',
                            'Turnover':'turnover'
                            }).sort_values('date').drop(['Unnamed: 11', 'bid', 'ask', 'Average price', 'trades'], axis=1)

    if isinstance(columns_to_add, dict):
        for key, value in columns_to_add.items():
            df[key] = value
    return df