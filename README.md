# Analise-de-acoes-com-Python
### Como obter os dados históricos e analisar ações da bolsa utilizando o Pandas

Esta é uma introdução de como analisar dados históricos de ações, índices e câmbios utilizando Python. A fonte utilizada foi o https://finance.yahoo.com, por onde é possível obter os dados de forma muito simples usando o Pandas. Este estudo não tem como finalidade a recomendação de compra de nenhum ativo, é somente uma demonstração de como utilizar a linguagem Python para começar uma análise de ativos na bolsa, sendo possível, a partir disso, analisar de forma mais profunda e implementar modelos preditivos de machine learning.

A importação dos dados é feita de forma muito simples, ao invés de fazer o download dos dados pelo site do yahoo nós podemos importar de forma personalizada as cotações históricas diretamente pelo método pandas_datareader.data.get_data_yahoo().

![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/tabela_yahoo.PNG)

Primeiro importei as cotações ajustadas de fechamento de pregão de ITUB3, BBAS3, BBDC3, SANB3 e do índice bovespa ^BVSP. O ticker correto a ser utilizado no código pode ser facilmente consultado no yahoo finance. Então puxei os dados de cotação ajustada por splits e dividendos (Adj Close) de 01/01/2008 até o 15/03/2020 (dia em que o estudo foi realizado). A fim de melhor visualização, as colunas foram renomeadas e foi realizada uma transformação na pontuação do Ibovespa, dividindo seus valores por 1000.

```python

from pandas_datareader import data as web

prices = pd.DataFrame()
tickers = ['ITUB3.SA', 'BBDC3.SA', 'BBAS3.SA', 'SANB3.SA', '^BVSP']
for i in tickers:
    prices[i] = web.get_data_yahoo(i,'01/01/2008')['Adj Close']
    
prices.rename(columns ={'ITUB3.SA':'ITUB', 'BBDC3.SA':'BBDC','BBAS3.SA':'BBAS','SANB3.SA':'SANB', '^BVSP':'IBOV'},inplace = True)
prices['IBOV'] = prices['IBOV']/1000
prices.reset_index(inplace = True)
prices.dropna(subset = ['IBOV'], inplace = True)
```
## 1) Cotação x tempo

```python
tickers = list(prices.drop(['Date'], axis = 1).columns)
plt.figure(figsize=(16,6))

for i in tickers:
        plt.plot(prices['Date'], prices[i])
plt.legend(tickers)
plt.grid()
plt.title("Cotação x tempo", fontsize = 25)
plt.show()
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/cotacaotempo.PNG)

Uma forma muito interessante de analisar o comportamento de um ativo no longo prazo é visualizar suas médias móveis, e isso pode ser feito de forma muito simples com o pacote Pandas. Utilizei o método .rolling do Pandas para criar e plotar médias móveis trimestrais e anuais de ITUB3, como exemplo.

```python
plt.figure(figsize=(16,6))
plt.plot(prices['Date'],prices['ITUB'].rolling(window = 90).mean())
plt.plot(prices['Date'], prices['ITUB'], alpha = 0.8)
plt.plot(prices['Date'],prices['ITUB'].rolling(window = 365).mean())
plt.grid()
plt.title('Cotações diárias e médias móveis de ITUB3', fontsize = 15)
plt.legend(['Média móvel trimestral','Cotação diária','Média móvel anual'])
plt.show()
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/mediasmóveis.PNG)

Então plotei um mapa de calor da correlação dos ativos. prices.corr() me retorna uma matriz numérica dos valores de correlação das cotações, mas fica muito mais agradável de visualizar esses dados em um mapa de calor. O parâmetro annot = True faz com que os valores fiquem visíveis no mapa.

```python
sns.heatmap(prices.corr(), annot = True)
plt.show()
```

![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/heatmap1.PNG)

## 2) Retorno diário

Com a cotação histórica também podemos criar um novo DataFrame de retorno diário, utilizando o método pct_change(). Com isso foi possível obter algumas informações mais profundas sobre os ativos utilizando os pacotes Pandas, Seaborn e Matplotlib. 

```python
returns = pd.DataFrame()
for i in tickers:
    returns[i] = prices[i].pct_change()
returns['Date'] = prices['Date']

returns.describe()
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/describe.PNG)

Esta simples tabela possui informações muito ricas, principalmente o desvio padrão e média dos retornos das ações. O desvio padrão(std) do retorno diário representa a volatilidade. O índice IBOV foi menos volátil que as ações dos 4 bancos neste período, mas também teve retorno diário médio baixo (mean = 0.000202).

Pairplot nos permite visualizar as relações entre cada variável do nosso dataset:

```python
sns.pairplot(returns)
plt.show()
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/pairplot1.PNG)

A distribuição das variações diárias do Ibovespa é uma distribuição normal:

```python
sns.distplot(returns['IBOV'].dropna())
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/distplot.PNG)

## 3) Retorno acumulado

O retorno acumulado de um ativo é calculado multiplicando os retornos diários somados por 1. Por exemplo: o retorno acumulado de 3 dias em que uma ação subiu 2% é (1.02 x 1.02 x 1.02) = 1.0612. Fiz este cálculo aplicando o método cumsum() no DataFrame de retornos diários. Este plot mostra o retorno acumulado desde 01/01/2008.

```python
return_sum = pd.DataFrame()
for ticker in tickers:
    return_sum[ticker] = (returns[ticker]+1).cumprod()
return_sum['Date'] = returns['Date']

plt.figure(figsize=(16,6))
plt.plot(return_sum['Date'], return_sum.drop(['Date'], axis = 1), alpha = 0.9)
plt.legend(tickers)
plt.title("Retorno x tempo", fontsize = 15)
plt.grid()
plt.show()
```
![screenshot1](https://github.com/matheuscoradini/Analise-de-acoes-com-Pandas/blob/master/imagens/retorno1.PNG)

## 4) Conclusão

O intuito deste pequeno estudo foi mostrar como é possível começar a fazer análise de dados de ações utilizando o Python. A partir disso é possível se aprofundar, obter insights valiosos, fazer análises preditivas e muito mais. Principalmente se você for um expert em mercado financeiro, o que não é meu caso :)
