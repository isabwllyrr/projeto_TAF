# Projeto TAF

Aplicativo em Streamlit para apoiar a analise quantitativa de ativos brasileiros. O projeto combina filtros de risco, indicadores fundamentalistas e modelos preditivos para transformar series financeiras em um painel de apoio a decisao.

Esta versao corresponde a **Entrega Final / Fase III**. O app integra a analise econometrica da Fase I, o motor preditivo da Fase II e a otimizacao de carteiras da Fase III em um fluxo unico de suporte a decisao.

## Proposta

O objetivo do app e ajudar o usuario a comparar ativos da B3 antes da composicao de uma carteira. Para isso, o painel organiza tres camadas de analise:

1. **Contexto do ativo**: dados fundamentalistas e comportamento historico.
2. **Risco estatistico**: CAPM, Fama-French e volatilidade condicional.
3. **Predicao**: modelos de machine learning e redes recorrentes para prever retornos futuros.
4. **Alocacao**: otimizacao de carteira com Markowitz, fronteira eficiente e HRP.

O foco da aplicacao e o mercado brasileiro. Por padrao, o benchmark usado e o Ibovespa (`^BVSP`) e os tickers seguem o formato do Yahoo Finance para ativos da B3, como `PETR4.SA`, `VALE3.SA` e `ITUB4.SA`.

## Principais recursos

- Selecao interativa de ativos brasileiros.
- Filtros por beta maximo e alfa anual minimo.
- Indicadores fundamentalistas obtidos via Yahoo Finance.
- Estimacao de beta, alfa e premio de risco pelo CAPM.
- Decomposicao de retornos com fatores de Fama-French, a partir de CSV de fatores brasileiros.
- Modelagem de volatilidade com GARCH(1,1).
- Previsao de retornos com Random Forest, Boosting e XGBoost.
- Validacao temporal com janela expansiva usando `sktime`.
- Backtesting da estrategia preditiva contra buy and hold.
- Estrutura de GRU e LSTM com PyTorch para capturar dependencias temporais.
- Otimizacao de carteiras pelo modelo Media-Variancia de Markowitz.
- Fronteira eficiente com restricao de peso maximo por ativo.
- Extensao de otimizacao por Hierarchical Risk Parity (HRP).
- Integracao entre previsoes da Fase II e retornos esperados usados na carteira final.
- Teste de estresse simples para avaliar o impacto de choques adversos na carteira.

## Estrutura do app

O dashboard esta dividido em abas:

- **Visao geral**: precos normalizados, retorno anual, volatilidade e drawdown.
- **Fundamentalista**: indicadores financeiros e contabeis dos ativos.
- **CAPM**: beta, alfa, premio de risco, R2 e significancia estatistica.
- **Fama-French**: exposicao aos fatores de mercado, tamanho e valor.
- **ARCH/GARCH**: volatilidade condicional e persistencia do risco.
- **Predicao**: metricas de erro, acuracia direcional e backtesting.
- **Otimizacao**: Markowitz, fronteira eficiente, HRP, pesos finais e teste de estresse.
- **Ranking**: consolidacao dos filtros para comparar os ativos.

## Como executar

No terminal, dentro da pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

O Streamlit exibira um endereco local, normalmente:

```text
http://localhost:8501
```

## Dados

Os precos e os indicadores fundamentalistas sao consultados no Yahoo Finance com `yfinance`.

O app nao usa dados simulados. Se a fonte online estiver indisponivel, e possivel enviar um CSV de precos reais pela barra lateral.

Exemplo de CSV de precos:

| date | PETR4.SA | VALE3.SA | ITUB4.SA | ^BVSP |
| --- | --- | --- | --- | --- |
| 2024-01-02 | 37.78 | 76.10 | 32.15 | 132697.00 |

Para o modelo de Fama-French, o app espera um CSV com fatores brasileiros:

| Coluna | Descricao |
| --- | --- |
| `date` | Data da observacao |
| `mkt-rf` | Retorno excedente do mercado |
| `smb` | Fator tamanho |
| `hml` | Fator valor |
| `rf` | Taxa livre de risco diaria, opcional |

## Observacoes

O app foi pensado como uma ferramenta de apoio analitico. Ele nao gera recomendacao automatica de compra ou venda; a ideia e organizar evidencias quantitativas para facilitar a comparacao entre ativos.

Na Fase III, o destaque esta na integracao entre predicao e alocacao. O usuario escolhe o peso das previsoes no retorno esperado, e o app mostra como essa escolha altera a composicao final da carteira. A carteira otimizada deixa de ser um modulo isolado e passa a ser consequencia dos filtros de risco e do motor preditivo.


