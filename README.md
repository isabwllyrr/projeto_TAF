# Projeto TAF - Entrega II

Web App em Streamlit para analise quantitativa de ativos brasileiros. O projeto evolui a Fase I, mantendo os filtros econometricos de risco e adicionando um motor preditivo com validacao temporal e backtesting.

O foco desta versao e trabalhar com dados do Brasil, usando ativos negociados na B3 e o Ibovespa como benchmark padrao.

## Funcionalidades

- **Selecao de ativos brasileiros**: lista de tickers da B3 com benchmark `^BVSP`.
- **Analise fundamentalista**: setor, valor de mercado, P/L, P/VP, ROE, margem liquida, dividend yield e divida/patrimonio.
- **CAPM**: beta, alfa, premio de risco, R2 e p-valor do beta.
- **Filtros por beta e alpha**: criterios explicitos para verificar se o ativo passa no filtro de risco.
- **Fama-French**: decomposicao dos retornos em mercado, tamanho (SMB) e valor (HML), usando CSV de fatores brasileiros.
- **ARCH/GARCH**: volatilidade condicional, persistencia da volatilidade e AIC.
- **Predicao e backtesting**: Random Forest, Boosting/XGBoost, validacao temporal com `sktime` e comparacao da estrategia contra buy and hold.
- **Deep Learning**: GRU e LSTM implementados com PyTorch para capturar dependencias temporais.

## Como executar

No terminal, dentro da pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Depois, acesse o endereco mostrado pelo Streamlit, normalmente:

```text
http://localhost:8501
```

## Como usar

1. Selecione os ativos brasileiros na barra lateral.
2. Confira o benchmark, que por padrao e o Ibovespa (`^BVSP`).
3. Defina o periodo historico e a taxa livre de risco.
4. Ajuste os filtros CAPM de beta maximo e alfa anual minimo.
5. Use as abas para analisar fundamentos, risco, predicao e backtesting.

## Dados

Os precos e os indicadores fundamentalistas sao obtidos no Yahoo Finance por meio da biblioteca `yfinance`.

O app nao utiliza dados simulados. Se a consulta online falhar, e necessario enviar um CSV com precos reais.

### CSV de precos reais

O CSV deve conter uma coluna de data e uma coluna para cada ativo, incluindo o benchmark:

| date | PETR4.SA | VALE3.SA | ITUB4.SA | ^BVSP |
| --- | --- | --- | --- | --- |
| 2024-01-02 | 37.78 | 76.10 | 32.15 | 132697.00 |

### CSV de fatores Fama-French

Para ativos brasileiros, o app espera fatores brasileiros. O CSV deve conter:

| Coluna | Descricao |
| --- | --- |
| `date` | Data da observacao |
| `mkt-rf` | Retorno excedente do mercado |
| `smb` | Fator tamanho |
| `hml` | Fator valor |
| `rf` | Taxa livre de risco diaria, opcional |

## Metodologia

O CAPM mede a sensibilidade dos ativos ao mercado e gera filtros por beta e alpha. O modelo de Fama-French acrescenta a decomposicao dos retornos por tamanho e valor. O GARCH modela a volatilidade condicional. Na Entrega II, o app adiciona modelos de aprendizado de maquina para prever retornos futuros e avalia essas previsoes com validacao cruzada para series temporais e backtesting.

O resultado e um painel de apoio a decisao: ele nao substitui uma recomendacao de investimento, mas organiza evidencias quantitativas para comparar ativos brasileiros.
