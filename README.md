# Analise-de-acoes-com-Pandas
Como obter os dados históricos e analisar ações da bolsa utilizando o Pandas

Esta é uma introdução de como analisar dados históricos de ações, índices e câmbios. A fonte utilizada foi o https://finance.yahoo.com, por onde é possível obter os dados de forma muito simples usando o pandas_datareader.data.get_data_yahoo(). Este estudo não tem como finalidade a recomendação de compra de nenhum ativo, é somente uma demonstração de como utilizar a linguagem Python para começar uma análise de ativos na bolsa, sendo possível, a partir disso, analisar de forma mais profunda e implementar modelos preditivos de machine learning.

A importação dos dados é feita de forma muito simples, ao invés de fazer o download dos dados pelo site do yahoo nós podemos importar de forma personalizada as cotações históricas diretamente pelo método pandas_datareader.data.get_data_yahoo().

-- imagem da tabela do yahoo

Neste estudo foram importadas as cotações ajustadas de fechamento de pregão de ITUB3, BBAS3, BBDC3, SANB3 e do índice bovespa ^BVSP. O ticker correto a ser utilizado no código pode ser facilmente consultado no yahoo finance. Foram puxados os dados de cotação ajustada por splits e dividendos (Adj Close) de 01/01/2008 até o 15/03/2020 (dia em que o estudo foi realizado). A fim de melhor visualização, as colunas foram renomeadas e foi realizada uma transformação na pontuação do Ibovespa, dividindo seus valores por 1000.

```python

from pandas_datareader import data as web

prices = pd.DataFrame()
tickers = ['ITUB3.SA', 'BBDC3.SA', 'BBAS3.SA', 'SANB3.SA', '^BVSP']
for i in tickers:
    prices[i] = web.get_data_yahoo(i,'01/01/2008')['Adj Close']
    
prices.rename(columns ={'ITUB3.SA':'ITUB', 'BBDC3.SA':'BBDC','BBAS3.SA':'BBAS','SANB3.SA':'SANB', '^BVSP':'IBOV'},inplace = True)
prices['IBOV'] = prices['IBOV']/1000
prices.reset_index(inplace = True)
```

-- imagem de cotação x tempo

