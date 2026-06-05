# Projeto TAF - Filtro Quantitativo de Ativos

Este projeto implementa um Web App em Streamlit para apoiar a primeira etapa de selecao de ativos. A ideia e reunir, em uma unica tela, indicadores fundamentalistas e medidas econometricas de risco para ajudar o usuario a comparar empresas antes de montar uma carteira.

O app foi desenvolvido para a Fase I da disciplina, com foco em analise de risco estatistico e econometria aplicada a series financeiras.

## O que o app faz

O usuario escolhe um mercado, seleciona os ativos e define o periodo da analise. A partir disso, o aplicativo baixa dados reais e organiza os resultados em abas:

- **Visao geral**: mostra a evolucao dos precos ajustados, retorno anual, volatilidade anual e drawdown maximo.
- **Fundamentalista**: resume setor, valor de mercado, P/L, P/VP, ROE, margem liquida, dividend yield e divida/patrimonio.
- **CAPM**: estima beta, alfa, premio de risco, R2 e p-valor do beta para cada ativo.
- **Fama-French**: decompõe os retornos nos fatores de mercado, tamanho (SMB) e valor (HML).
- **ARCH/GARCH**: estima a volatilidade condicional e a persistencia da volatilidade.
- **Ranking**: combina retorno, volatilidade, drawdown, beta e risco condicional em um filtro final de risco relativo.

## Fontes de dados

Os precos e indicadores fundamentalistas sao obtidos pelo Yahoo Finance usando a biblioteca `yfinance`.

Os fatores de Fama-French sao buscados automaticamente na Kenneth French Data Library por meio da biblioteca `pandas-datareader`. Caso a fonte automatica nao esteja disponivel, o app tambem aceita um CSV manual com os fatores.

O projeto nao utiliza dados simulados. Se as fontes online falharem, e necessario enviar um CSV com dados reais.

## Como executar

No terminal, dentro da pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Depois, acesse o endereco exibido pelo Streamlit, normalmente:

```text
http://localhost:8501
```

## Como usar

1. Escolha o mercado na barra lateral.
2. Selecione os ativos pela lista.
3. Confira ou altere o benchmark.
4. Defina o periodo historico.
5. Ajuste a taxa livre de risco anual.
6. Analise os resultados nas abas do dashboard.

Para ativos fora das listas sugeridas, use a opcao avancada de tickers manuais.

## CSV de precos reais

Se for necessario carregar precos manualmente, o CSV deve conter uma coluna de data e uma coluna para cada ativo, incluindo o benchmark.

Exemplo:

| date | AAPL | MSFT | NVDA | ^GSPC |
| --- | --- | --- | --- | --- |
| 2024-01-02 | 185.64 | 368.85 | 48.17 | 4742.83 |

## CSV de fatores Fama-French

O CSV de fatores deve conter:

| Coluna | Descricao |
| --- | --- |
| `date` | Data da observacao |
| `mkt-rf` | Retorno excedente do mercado |
| `smb` | Fator tamanho |
| `hml` | Fator valor |
| `rf` | Taxa livre de risco diaria, opcional |

Os fatores podem estar em formato decimal (`0.01`) ou percentual (`1.0`).

## Observacao metodologica

O CAPM usa os retornos excedentes do ativo e do benchmark para estimar a sensibilidade ao mercado. O modelo de Fama-French amplia essa analise ao separar os efeitos de mercado, tamanho e valor. Ja o GARCH ajuda a observar a dinamica da volatilidade ao longo do tempo.

Com isso, o app funciona como um primeiro filtro: ele nao decide automaticamente quais ativos comprar, mas organiza evidencias quantitativas para apoiar a comparacao entre alternativas.
